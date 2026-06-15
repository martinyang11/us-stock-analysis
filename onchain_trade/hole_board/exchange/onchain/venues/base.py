from abc import ABC, abstractmethod

from hole_board.exchange.onchain.types import OnchainCloseRequest, OnchainConfig, OnchainOpenRequest, OnchainTradeResult


class OnchainVenueAdapter(ABC):
    def __init__(self, config: OnchainConfig):
        self.config = config

    @abstractmethod
    def open_trade(self, req: OnchainOpenRequest) -> OnchainTradeResult:
        raise NotImplementedError

    @abstractmethod
    def close_trade(self, req: OnchainCloseRequest) -> OnchainTradeResult:
        raise NotImplementedError

    @abstractmethod
    def fetch_positions(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def fetch_wallet_balance(self, token: str) -> float:
        raise NotImplementedError
