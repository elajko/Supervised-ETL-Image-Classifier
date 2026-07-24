import re

import httpx

from app.sources.base import SourceAdapter, SourceImage

_ALBUM_RE = re.compile(r"imgur\.com/a/([a-zA-Z0-9]+)")
_GALLERY_RE = re.compile(r"imgur\.com/gallery/([a-zA-Z0-9]+)")
_USER_RE = re.compile(r"imgur\.com/user/([^/]+)")

# Safety cap on paginated submission fetches (20 pages * ~30 posts/page is
# generous for a single crawl run without risking an unbounded loop against
# a very prolific account).
_MAX_USER_PAGES = 20


class ImgurAdapter(SourceAdapter):
    """Imgur's public API only requires an app-registered Client-ID (no
    per-user login) to read public albums, gallery posts, and a user's
    public submissions -- so this adapter never needs the interactive
    auth-tab flow, just the one credential."""

    name = "imgur"
    domains = ("imgur.com",)
    needs_client_secret = False
    supports_interactive_auth = False

    async def is_authenticated(self) -> bool:
        return await self.is_configured()

    async def fetch_images(self, seed_url: str) -> list[SourceImage]:
        creds = await self.get_credentials()
        if not creds or not creds.get("client_id"):
            raise RuntimeError("Imgur Client ID is not configured")
        headers = {"Authorization": f"Client-ID {creds['client_id']}"}
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            album_match = _ALBUM_RE.search(seed_url)
            if album_match:
                return await self._fetch_album(client, album_match.group(1), seed_url)
            gallery_match = _GALLERY_RE.search(seed_url)
            if gallery_match:
                return await self._fetch_gallery(client, gallery_match.group(1), seed_url)
            user_match = _USER_RE.search(seed_url)
            if user_match:
                return await self._fetch_user(client, user_match.group(1), seed_url)
            raise ValueError(f"unrecognized Imgur URL format: {seed_url}")

    @staticmethod
    async def _fetch_album(client: httpx.AsyncClient, album_hash: str, seed_url: str) -> list[SourceImage]:
        resp = await client.get(f"https://api.imgur.com/3/album/{album_hash}/images")
        resp.raise_for_status()
        items = resp.json()["data"]
        return [SourceImage(url=item["link"], source_page_url=seed_url) for item in items if not item.get("animated")]

    @staticmethod
    async def _fetch_gallery(client: httpx.AsyncClient, gallery_hash: str, seed_url: str) -> list[SourceImage]:
        resp = await client.get(f"https://api.imgur.com/3/gallery/{gallery_hash}")
        resp.raise_for_status()
        item = resp.json()["data"]
        images = item.get("images") if item.get("is_album") else [item]
        return [SourceImage(url=i["link"], source_page_url=seed_url) for i in images if not i.get("animated")]

    @staticmethod
    async def _fetch_user(client: httpx.AsyncClient, username: str, seed_url: str) -> list[SourceImage]:
        results: list[SourceImage] = []
        for page in range(_MAX_USER_PAGES):
            resp = await client.get(f"https://api.imgur.com/3/account/{username}/submissions/{page}")
            resp.raise_for_status()
            posts = resp.json()["data"]
            if not posts:
                break
            for post in posts:
                if post.get("images"):
                    results.extend(
                        SourceImage(url=i["link"], source_page_url=seed_url)
                        for i in post["images"]
                        if not i.get("animated")
                    )
                elif post.get("link"):
                    results.append(SourceImage(url=post["link"], source_page_url=seed_url))
        return results
