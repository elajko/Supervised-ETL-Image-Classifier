from typing import Optional

from app.sources.base import SourceAdapter
from app.sources.deviantart import DeviantArtAdapter
from app.sources.imgur import ImgurAdapter
from app.sources.pinterest import PinterestAdapter

ADAPTERS: list[SourceAdapter] = [ImgurAdapter(), DeviantArtAdapter(), PinterestAdapter()]


def match_source(url: str) -> Optional[SourceAdapter]:
    for adapter in ADAPTERS:
        if adapter.matches(url):
            return adapter
    return None


def get_adapter(name: str) -> Optional[SourceAdapter]:
    for adapter in ADAPTERS:
        if adapter.name == name:
            return adapter
    return None
