import hashlib
import io
from typing import Optional

import httpx
from PIL import Image
from playwright.async_api import Page


# Many gallery sites serve a downscaled preview by default (to save bandwidth)
# and link the full-resolution original elsewhere on the same element: a
# srcset with larger descriptors, a lazy-load data-* attribute, or a wrapping
# <a> whose href points straight at the full image file. Prefer those over
# the rendered <img src> so we don't collect throwaway thumbnails.
_BEST_IMAGE_URL_JS = r"""
els => els.map(img => {
  function resolve(url) {
    try { return new URL(url, document.baseURI).href; } catch (e) { return null; }
  }
  const lazyAttrs = ['data-src', 'data-original', 'data-full', 'data-large-src', 'data-lazy-src', 'data-file-url', 'data-large-file-url'];
  for (const attr of lazyAttrs) {
    const v = img.getAttribute(attr);
    if (v) {
      const resolved = resolve(v);
      if (resolved) return resolved;
    }
  }
  const srcset = img.getAttribute('srcset');
  if (srcset) {
    const entries = srcset.split(',').map(s => s.trim()).filter(Boolean).map(s => {
      const parts = s.split(/\s+/);
      const descriptor = parts[1] || '';
      const width = descriptor.endsWith('w') ? parseInt(descriptor, 10) : 0;
      return { url: parts[0], width };
    });
    entries.sort((a, b) => b.width - a.width);
    if (entries.length) {
      const resolved = resolve(entries[0].url);
      if (resolved) return resolved;
    }
  }
  const parentA = img.closest('a');
  if (parentA && parentA.href && /\.(jpe?g|png|gif|webp|bmp)(\?.*)?$/i.test(parentA.href)) {
    return parentA.href;
  }
  return img.src || null;
}).filter(Boolean)
"""


async def extract_image_urls(page: Page) -> list[str]:
    srcs = await page.eval_on_selector_all("img", _BEST_IMAGE_URL_JS)
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
        # Some sites serve certain images (e.g. rarely-accessed legacy
        # thumbnails behind a CDN cold cache) slowly and unreliably --
        # observed hanging for ~5s before the origin resets the connection.
        # A shorter timeout means we give up on those faster rather than
        # tying up a concurrency slot waiting for a request likely to fail
        # anyway; genuinely slow-but-working connections still have a
        # generous window.
        resp = await client.get(url, timeout=8)
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


def meets_minimum_size(image_bytes: bytes, min_width: int, min_height: int) -> bool:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except Exception:
        return False
    return width >= min_width and height >= min_height
