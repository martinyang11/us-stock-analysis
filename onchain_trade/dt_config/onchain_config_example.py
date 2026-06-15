"""
链上交易配置示例。复制为 onchain_config.py 并填入真实值（勿提交 git）。
"""
# Gains / gTrade 合约地址（按目标链填写）
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
