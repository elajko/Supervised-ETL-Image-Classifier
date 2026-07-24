import re
import time
import urllib.parse
from typing import Optional

import httpx

from app import db
from app.sources.base import SourceAdapter, SourceImage

_PROFILE_RE = re.compile(r"deviantart\.com/([a-zA-Z0-9_-]+)")
TOKEN_URL = "https://www.deviantart.com/oauth2/token"
AUTHORIZE_URL = "https://www.deviantart.com/oauth2/authorize"
API_BASE = "https://www.deviantart.com/api/v1/oauth2"

# Safety cap on paginated gallery fetches (80 pages * 24 deviations/page).
_MAX_GALLERY_PAGES = 80


class DeviantArtAdapter(SourceAdapter):
    """DeviantArt's client-credentials grant (app-only, no per-user login)
    is enough to browse most public galleries, so that's tried first and
    automatically. The interactive auth-tab flow is offered as an optional
    upgrade -- if a specific gallery still rejects the app-only token (e.g.
    it's mature-content-gated), `fetch_images` raises `PermissionError` and
    the caller can prompt for interactive login then."""

    name = "deviantart"
    domains = ("deviantart.com",)
    needs_client_secret = True
    supports_interactive_auth = True

    async def is_authenticated(self) -> bool:
        return await self._get_valid_token() is not None

    async def _get_valid_token(self) -> Optional[str]:
        row = await db.get_oauth_token(self.name)
        if row and row.get("expires_at") and time.time() < float(row["expires_at"]):
            return row["access_token"]

        creds = await self.get_credentials()
        if not creds or not creds.get("client_id") or not creds.get("client_secret"):
            return None
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": creds["client_id"],
                    "client_secret": creds["client_secret"],
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            expires_at = time.time() + data.get("expires_in", 3600) - 60
            await db.set_oauth_token(
                self.name, data["access_token"], row["refresh_token"] if row else None, str(expires_at)
            )
            return data["access_token"]

    def get_auth_url(self, redirect_uri: str, credentials: dict) -> str:
        params = {
            "response_type": "code",
            "client_id": credentials["client_id"],
            "redirect_uri": redirect_uri,
            "scope": "browse",
            "state": self.name,
        }
        return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    async def handle_callback(self, code: str, redirect_uri: str) -> None:
        creds = await self.get_credentials()
        if not creds:
            raise RuntimeError("DeviantArt credentials are not configured")
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": creds["client_id"],
                    "client_secret": creds["client_secret"],
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            expires_at = time.time() + data.get("expires_in", 3600) - 60
            await db.set_oauth_token(self.name, data["access_token"], data.get("refresh_token"), str(expires_at))

    async def fetch_images(self, seed_url: str) -> list[SourceImage]:
        token = await self._get_valid_token()
        if not token:
            raise RuntimeError("DeviantArt authentication required")
        match = _PROFILE_RE.search(seed_url)
        if not match:
            raise ValueError(f"unrecognized DeviantArt URL format: {seed_url}")
        username = match.group(1)

        images: list[SourceImage] = []
        offset = 0
        async with httpx.AsyncClient(timeout=15.0) as client:
            for _ in range(_MAX_GALLERY_PAGES):
                resp = await client.get(
                    f"{API_BASE}/gallery/all",
                    params={
                        "username": username,
                        "offset": offset,
                        "limit": 24,
                        "mature_content": "false",
                        "access_token": token,
                    },
                )
                if resp.status_code in (401, 403):
                    raise PermissionError(
                        "DeviantArt rejected the app-only token for this gallery -- "
                        "interactive login is required for this content"
                    )
                resp.raise_for_status()
                data = resp.json()
                for deviation in data.get("results", []):
                    content = deviation.get("content")
                    if content and content.get("src"):
                        images.append(SourceImage(url=content["src"], source_page_url=seed_url))
                if not data.get("has_more"):
                    break
                offset = data.get("next_offset", offset + 24)
        return images
