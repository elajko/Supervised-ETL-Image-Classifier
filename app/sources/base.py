from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from app import db


@dataclass
class SourceImage:
    url: str
    source_page_url: str


class SourceAdapter:
    """Base class for site-specific, API-based image sources that bypass
    the generic Playwright scraper -- for sites whose terms forbid bulk
    scraping, or that require an authenticated session the crawler can't
    reach on its own.

    Credentials are app-level (a client id/secret the *developer* registers
    with the site, stored in `source_credentials`) and are distinct from
    `oauth_tokens`, which holds the per-user access token obtained via the
    interactive login-tab flow, if the adapter supports one.
    """

    name: str
    domains: tuple[str, ...] = ()
    needs_client_secret: bool = False
    # Whether an "Authenticate with X" login-tab flow exists for this site at
    # all. Some adapters (Imgur) never need it; others (DeviantArt) offer it
    # as an optional upgrade path beyond app-only access; others (Pinterest)
    # require it before any request can succeed.
    supports_interactive_auth: bool = False

    def matches(self, url: str) -> bool:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return any(netloc == d or netloc.endswith("." + d) for d in self.domains)

    async def get_credentials(self) -> Optional[dict]:
        return await db.get_source_credentials(self.name)

    async def is_configured(self) -> bool:
        creds = await self.get_credentials()
        return bool(creds and creds.get("client_id"))

    async def is_authenticated(self) -> bool:
        """Whether this adapter is ready to fetch images right now."""
        raise NotImplementedError

    def get_auth_url(self, redirect_uri: str, credentials: dict) -> str:
        raise NotImplementedError

    async def handle_callback(self, code: str, redirect_uri: str) -> None:
        raise NotImplementedError

    async def fetch_images(self, seed_url: str) -> list[SourceImage]:
        raise NotImplementedError
