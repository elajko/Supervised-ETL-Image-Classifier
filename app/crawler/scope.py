from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


def in_scope(url: str, seed_netlocs: set[str]) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc in seed_netlocs


class RobotsCache:
    """Fetches and caches robots.txt per origin. Fails open (allows fetching)
    if robots.txt is missing or unreachable."""

    def __init__(self) -> None:
        self._cache: dict[str, RobotFileParser] = {}

    async def is_allowed(self, client: httpx.AsyncClient, url: str, user_agent: str = "*") -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._cache.get(origin)
        if rp is None:
            rp = RobotFileParser()
            try:
                resp = await client.get(f"{origin}/robots.txt", timeout=10)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.parse([])
            except httpx.HTTPError:
                rp.parse([])
            self._cache[origin] = rp
        return rp.can_fetch(user_agent, url)
