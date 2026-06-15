"""
链上交易配置。勿提交 git。
"""
# ⚠️ 填入你的 Arbitrum 钱包地址和私钥
WALLET_ADDRESS = "0x0000000000000000000000000000000000000000"
PRIVATE_KEY = "0x0000000000000000000000000000000000000000000000000000000000000000"

# Gains / gTrade 合约地址
GAINS_CONTRACTS = {
    "trading_router": "0x0000000000000000000000000000000000000000",
    "vault": "0x0000000000000000000000000000000000000000",
}

ONCHAIN_DEFAULTS = {
    "onchain_venue": "gains",
    "chain_id": 42161,
    "rpc_url": "https://arb1.arbitrum.io/rpc",
    "dry_run": True,
    "gains_contracts": GAINS_CONTRACTS,
}
