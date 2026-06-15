"""从 aitrader Account 字段解析链上配置。"""
from __future__ import annotations

from hole_board.exchange.onchain.types import OnchainConfig, WalletSource

API_NAME_TO_WALLET_SOURCE: dict[str, WalletSource] = {
    "binance_onchain": "binance",
    "okx_onchain": "okx",
}

DEFAULT_RPC_BY_CHAIN: dict[int, str] = {
    42161: "https://arb1.arbitrum.io/rpc",
    8453: "https://mainnet.base.org",
}


def parse_onchain_config(
    api_name: str,
    user_id: str,
    password: str,
    *,
    onchain_venue: str = "gains",
    chain_id: int = 42161,
    rpc_url: str | None = None,
    dry_run: bool = True,
    gains_contracts: dict | None = None,
) -> OnchainConfig:
    wallet_source = API_NAME_TO_WALLET_SOURCE.get(api_name, "external")
    return OnchainConfig(
        wallet_address=user_id.strip(),
        private_key=password.strip(),
        wallet_source=wallet_source,
        onchain_venue=onchain_venue,
        chain_id=chain_id,
        rpc_url=rpc_url or DEFAULT_RPC_BY_CHAIN.get(chain_id, ""),
        dry_run=dry_run,
        gains_contracts=gains_contracts or {},
    )
