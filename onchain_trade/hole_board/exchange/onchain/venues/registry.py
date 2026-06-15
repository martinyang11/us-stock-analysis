from hole_board.exchange.onchain.types import OnchainConfig
from hole_board.exchange.onchain.venues.base import OnchainVenueAdapter
from hole_board.exchange.onchain.venues.gains.adapter import GainsVenueAdapter

VENUE_ADAPTERS: dict[str, type[OnchainVenueAdapter]] = {
    "gains": GainsVenueAdapter,
}


def get_venue_adapter(config: OnchainConfig) -> OnchainVenueAdapter:
    venue = (config.onchain_venue or "gains").lower()
    cls = VENUE_ADAPTERS.get(venue)
    if cls is None:
        raise ValueError(f"未知 onchain_venue: {venue}，已注册: {list(VENUE_ADAPTERS)}")
    return cls(config)
