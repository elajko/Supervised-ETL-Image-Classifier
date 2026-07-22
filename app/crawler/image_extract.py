import hashlib
from typing import Optional

import httpx
from playwright.async_api import Page


async def extract_image_urls(page: Page) -> list[str]:
    srcs = await page.eval_on_selector_all("img", "els => els.map(e => e.src)")
    return _dedupe_preserve_order(s for s in srcs if s)


async def extract_link_urls(page: Page) -> list[str]:
    hrefs = await page.eval_on_selector_all("a", "els => els.map(e => e.href)")
    return _dedupe_preserve_order(h for h in hrefs if h)


def _dedupe_preserve_order(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


async def fetch_image_bytes(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    try:
        resp = await client.get(url, timeout=15)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    content_type = resp.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        return None
    return resp.content


def content_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()
