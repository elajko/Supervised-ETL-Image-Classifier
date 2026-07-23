import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx
import numpy as np
from playwright.async_api import async_playwright

from app.config import (
    MAX_CONCURRENT_EMBEDDINGS,
    MAX_CONCURRENT_IMAGE_FETCHES,
    MAX_CONCURRENT_PAGES,
    MAX_DEPTH,
    MAX_PAGES_PER_SESSION,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)
from app.crawler.image_extract import content_hash, extract_image_urls, extract_link_urls, fetch_image_bytes, meets_minimum_size
from app.crawler.scope import RobotsCache, in_scope
from app.ml.embeddings import embed_image


@dataclass
class CrawledImage:
    content_hash: str
    image_url: str
    source_page_url: str
    image_bytes: bytes
    embedding: np.ndarray


Sink = Callable[[CrawledImage], Awaitable[None]]
IsDuplicate = Callable[[str], Awaitable[bool]]
OnProgress = Callable[..., Awaitable[None]]


class Crawler:
    """BFS crawler over seed URLs + same-domain linked pages (default depth 1).
    For each in-scope page, extracts <img> sources, downloads bytes, dedupes by
    content hash, computes a CLIP embedding, and hands the result to `sink`."""

    def __init__(
        self,
        seed_urls: list[str],
        is_duplicate: IsDuplicate,
        sink: Sink,
        on_progress: Optional[OnProgress] = None,
        stop_event: Optional[asyncio.Event] = None,
        max_depth: int = MAX_DEPTH,
        max_pages: int = MAX_PAGES_PER_SESSION,
        max_concurrent: int = MAX_CONCURRENT_PAGES,
    ) -> None:
        self.seed_urls = seed_urls
        self.is_duplicate = is_duplicate
        self.sink = sink
        self.on_progress = on_progress
        self.stop_event = stop_event or asyncio.Event()
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.image_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGE_FETCHES)
        self.embed_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDINGS)

        self.seed_netlocs = {urlparse(u).netloc for u in seed_urls}
        self.visited: set[str] = set()
        self.pages_visited = 0
        self.images_found = 0

    async def run(self) -> None:
        robots = RobotsCache()
        async with httpx.AsyncClient(follow_redirects=True) as http_client:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    wave = [(u, 0) for u in self.seed_urls]
                    while wave and not self.stop_event.is_set() and self.pages_visited < self.max_pages:
                        results = await asyncio.gather(
                            *(self._visit(browser, http_client, robots, url, depth) for url, depth in wave)
                        )
                        wave = [link for links in results for link in links]
                finally:
                    await browser.close()

    async def _visit(self, browser, http_client: httpx.AsyncClient, robots: RobotsCache, url: str, depth: int):
        if self.stop_event.is_set() or url in self.visited or self.pages_visited >= self.max_pages:
            return []
        self.visited.add(url)

        if not in_scope(url, self.seed_netlocs):
            return []
        if not await robots.is_allowed(http_client, url):
            return []

        async with self.semaphore:
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                img_srcs = await extract_image_urls(page)
                hrefs = await extract_link_urls(page) if depth < self.max_depth else []
            except Exception:
                return []
            finally:
                await page.close()

        self.pages_visited += 1
        if self.on_progress:
            await self.on_progress(pages_visited=self.pages_visited, current_url=url)

        await self._process_images(http_client, img_srcs, url)

        if self.stop_event.is_set():
            return []

        next_links = []
        for href in hrefs:
            if href not in self.visited and in_scope(href, self.seed_netlocs):
                next_links.append((href, depth + 1))
        return next_links

    async def _process_images(self, http_client: httpx.AsyncClient, img_srcs: list[str], source_page_url: str) -> None:
        await asyncio.gather(*(self._process_one_image(http_client, src, source_page_url) for src in img_srcs))

    async def _process_one_image(self, http_client: httpx.AsyncClient, src: str, source_page_url: str) -> None:
        # Isolated per image and bounded by image_semaphore: a single page can
        # reference hundreds of <img> tags (e.g. a forum's thread-list sidebar
        # nav, present on every page), and gather() would otherwise cancel
        # every sibling task the moment one of them raises.
        if self.stop_event.is_set():
            return
        try:
            async with self.image_semaphore:
                image_bytes = await fetch_image_bytes(http_client, src)
            if image_bytes is None:
                return
            if not meets_minimum_size(image_bytes, MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT):
                return
            chash = content_hash(image_bytes)
            if await self.is_duplicate(chash):
                return
            async with self.embed_semaphore:
                embedding = await asyncio.to_thread(embed_image, image_bytes)
            self.images_found += 1
            if self.on_progress:
                await self.on_progress(images_found=self.images_found)
            await self.sink(CrawledImage(chash, src, source_page_url, image_bytes, embedding))
        except Exception:
            return
