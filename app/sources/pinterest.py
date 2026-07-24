import re
import urllib.parse
from typing import Optional

import httpx

from app import db
from app.sources.base import SourceAdapter, SourceImage

_BOARD_RE = re.compile(r"pinterest\.[a-z.]+/([^/]+)/([^/]+)/?$")
AUTHORIZE_URL = "https://www.pinterest.com/oauth/"
TOKEN_URL = "https://api.pinterest.com/v5/oauth/token"
API_BASE = "https://api.pinterest.com/v5"
SCOPE = "boards:read,pins:read"

_MAX_PAGES = 40  # safety cap on paginated board/pin listings


class PinterestAdapter(SourceAdapter):
    """Pinterest's v5 API is scoped almost entirely to the *authenticated
    user's own* boards and pins -- there's no supported way to anonymously
    browse or search someone else's public boards through the official API
    anymore. So this only works for boards the logged-in Pinterest account
    owns or follows, not arbitrary topic scraping, and always requires the
    interactive login-tab flow."""

    name = "pinterest"
    domains = ("pinterest.com",)
    needs_client_secret = True
    supports_interactive_auth = True

    async def is_authenticated(self) -> bool:
        row = await db.get_oauth_token(self.name)
        return row is not None

    def get_auth_url(self, redirect_uri: str, credentials: dict) -> str:
        params = {
            "response_type": "code",
            "client_id": credentials["client_id"],
            "redirect_uri": redirect_uri,
            "scope": SCOPE,
            "state": self.name,
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def handle_callback(self, code: str, redirect_uri: str) -> None:
        creds = await self.get_credentials()
        if not creds:
            raise RuntimeError("Pinterest credentials are not configured")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                TOKEN_URL,
                data={"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri},
                auth=(creds["client_id"], creds["client_secret"]),
            )
            resp.raise_for_status()
            data = resp.json()
            await db.set_oauth_token(self.name, data["access_token"], data.get("refresh_token"), None)

    async def _get_token(self) -> Optional[str]:
        row = await db.get_oauth_token(self.name)
        return row["access_token"] if row else None

    async def fetch_images(self, seed_url: str) -> list[SourceImage]:
        token = await self._get_token()
        if not token:
            raise RuntimeError("Pinterest authentication required")
        match = _BOARD_RE.search(seed_url)
        if not match:
            raise ValueError(f"unrecognized Pinterest board URL: {seed_url}")
        _owner, board_slug = match.groups()

        headers = {"Authorization": f"Bearer {token}"}
        images: list[SourceImage] = []
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            board_id = await self._find_board_id(client, board_slug)
            if board_id is None:
                raise ValueError(
                    "Pinterest's API only exposes boards owned by the authenticated "
                    "account -- this board wasn't found among them"
                )
            bookmark = None
            for _ in range(_MAX_PAGES):
                params: dict = {"page_size": 25}
                if bookmark:
                    params["bookmark"] = bookmark
                resp = await client.get(f"{API_BASE}/boards/{board_id}/pins", params=params)
                resp.raise_for_status()
                data = resp.json()
                for pin in data.get("items", []):
                    url = self._best_image_url(pin)
                    if url:
                        images.append(SourceImage(url=url, source_page_url=seed_url))
                bookmark = data.get("bookmark")
                if not bookmark:
                    break
        return images

    async def _find_board_id(self, client: httpx.AsyncClient, board_slug: str) -> Optional[str]:
        bookmark = None
        for _ in range(_MAX_PAGES):
            params: dict = {"page_size": 25}
            if bookmark:
                params["bookmark"] = bookmark
            resp = await client.get(f"{API_BASE}/boards", params=params)
            resp.raise_for_status()
            data = resp.json()
            for board in data.get("items", []):
                if board.get("name", "").strip().lower().replace(" ", "-") == board_slug.lower():
                    return board["id"]
            bookmark = data.get("bookmark")
            if not bookmark:
                return None
        return None

    @staticmethod
    def _best_image_url(pin: dict) -> Optional[str]:
        images = pin.get("media", {}).get("images", {})
        for key in ("originals", "1200x", "600x", "400x300"):
            if key in images and images[key].get("url"):
                return images[key]["url"]
        for value in images.values():
            if isinstance(value, dict) and value.get("url"):
                return value["url"]
        return None
