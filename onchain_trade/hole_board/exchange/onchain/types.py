from dataclasses import dataclass, field
from typing import Any, Literal, Optional

WalletSource = Literal["binance", "okx", "external"]


@dataclass
class OnchainConfig:
    wallet_address: str
    private_key: str
    wallet_source: WalletSource = "external"
    onchain_venue: str = "gains"
    chain_id: int = 42161
    rpc_url: str = ""
    dry_run: bool = True
    gains_contracts: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnchainOpenRequest:
    symbol: str
    side: Literal["long", "short"] = "long"
    collateral: float = 0.0
    leverage: int = 2
    slippage: float = 0.01


@dataclass
class OnchainCloseRequest:
    symbol: str
    position_id: str | None = None
    close_ratio: float = 1.0
    slippage: float = 0.01  # 平仓滑点容忍度，默认 1%


@dataclass
class OnchainTradeResult:
    tx_hash: str | None
    order_sys_id: str
    symbol: str
    side: str
    status: str
    raw: dict[str, Any] = field(default_factory=dict)
