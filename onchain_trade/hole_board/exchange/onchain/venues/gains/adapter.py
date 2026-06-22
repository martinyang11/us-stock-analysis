"""
Gains / gTrade 链上交易场所适配器。
合约地址与 ABI 通过 OnchainConfig.gains_contracts 注入；dry_run=True 时不发链上交易。
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Optional

from hole_board.exchange.onchain.types import OnchainCloseRequest, OnchainConfig, OnchainOpenRequest, OnchainTradeResult
from hole_board.exchange.onchain.venues.base import OnchainVenueAdapter
try:
    from hole_board.utils.log import new_logger
except ImportError:
    import logging

    def new_logger(name):
        return logging.getLogger(name)


if TYPE_CHECKING:
    from web3 import Web3
    from web3.types import TxReceipt

_logger = new_logger("GainsVenueAdapter")

# GNSMultiCollatDiamond on Arbitrum
DIAMOND_ADDRESS = "0xFF162c694eAA571f685030649814282eA457f169"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# ── Gas / Priority 可调参数 ───────────────────────────────────────────
# 修改这些值即可快速测试不同 gas 策略，无需到处改代码。
GAS_LIMIT_OPEN_TRADE = 2_000_000      # open_trade 交易的 gas limit
GAS_LIMIT_CLOSE_TRADE = 2_000_000     # close_trade 交易的 gas limit
MIN_PRIORITY_FEE_GWEI = 0           # maxPriorityFeePerGas 最低底线 (gwei)
# ─────────────────────────────────────────────────────────────────────

# Chainlink 最小 ABI（查询 latestRoundData）
_CHAINLINK_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"internalType": "uint80", "name": "roundId", "type": "uint80"},
            {"internalType": "int256", "name": "answer", "type": "int256"},
            {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
            {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
            {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Gains pairIndex → Chainlink feed 地址（Arbitrum）
# 自动生成自：https://github.com/thevolcanomanishere/gud-price/blob/main/FEEDS.md
_CHAINLINK_FEEDS: dict[int, str] = {
    0: "0x6ce185860a4963106506C203335A2910413708e9",   # BTC
    1: "0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612",   # ETH
    2: "0x86E53CF1B870786351Da77A57575e79CB55812CB",   # LINK
    3: "0x9A7FB1b3950837a8D9b40517626E11D4127C098C",   # DOGE
    5: "0xD9f615A9b820225edbA2d821c4A696a0924051c6",   # ADA
    7: "0xaD1d5344AaDE45F43E596773Bcc4c423EAbdD034",   # AAVE
    10: "0xe7C53FFd03Eb6ceF7d208bC4C13446c76d1E5884",   # COMP
    11: "0xa6bC5bAF2000424e90434bA7104ee399dEe80DEc",   # DOT
    17: "0x9C917083fDb403ab5ADbEC26Ee294f6EcAda2720",   # UNI
    19: "0xB4AD57B52aB9141de9926a3e0C8dc6264c2ef205",   # XRP
    20: "0x21082CA28570f0ccfb089465bFaEfDc77b00D367",   # ZEC
    21: "0xA14d53bC1F1c0F31B4aA3BD109344E5009051a84",   # EUR (EUR/USD)
    23: "0x9C4424Fd84C6661F97D8d6b3fc3C1aAc2BeDd137",   # GBP (GBP/USD)
    33: "0x24ceA4b8ce57cdA5058b924B9B9987992450590c",   # SOL
    34: "0xf3451Fd5eddE08cDAE95A6233BaD69DE95552a61",   # XTZ
    37: "0xaebDA2c976cfd1eE1977Eac079B4382acb849325",   # CRV
    47: "0x6970460aabF80C5BE983C6b74e5D06dEDCA95D4A",   # BNB
    49: "0x0F38D86FceF4955B705F35c9e41d1A16e0637c73",   # GRT
    57: "0x0E278D14B4bf6429dDB0a1B353e2Ae8A4e128C93",   # SHIB
    58: "0x8d0CC5f38f9E802475f2CFf4F9fc7000C2E1557c",   # AAPL
    62: "0xDde33fb9F21739602806580bdd73BAd831DcA867",   # MSFT
    65: "0x4881A4418b5F2460B21d6F08CD5aA0678a7f262F",   # NVDA
    81: "0xcd1bd86fDc33080DCF1b5715B6FCe04eC6F85845",   # META
    82: "0x1D1a83331e9D255EB1Aaf75026B60dFD00A252ba",   # GOOGL
    84: "0xd6a77691f071E98Df7217BED98f38ae6d2313EBA",   # AMZN
    85: "0x3609baAa0a9b1f0FE4d6CC01884585d0e191C3E3",   # TSLA
    86: "0x46306F3795342117721D8DEd50fbcF6DF2b3cc10",   # SPY
    90: "0x1F954Dc24a49708C26E0C1777f16750B5C6d5a2c",   # XAU
    91: "0xC56765f04B248394CF1619D20dB8082Edbfa75b1",   # XAG
    102: "0x8bf61728eeDCE2F32c456454d87B5d6eD6150208",   # AVAX
    103: "0xCDA67618e51762235eacA373894F0C79256768fa",   # ATOM
    104: "0xBF5C3fB2633e924598A46B9D07a174a9DBcF57C0",   # NEAR
    107: "0x0301e5D0A8f7490444ebd1921E3d0f0fe7722786",   # TON
    109: "0xb2A824043730FE05F3DA2efaFa1CBbe83fa548D6",   # ARB
    128: "0xA43A34030088E6510FecCFb77E88ee5e7ed0fE64",   # LDO
    131: "0x256654437f1ADA8057684b18d742eFD14034C400",   # CAKE
    134: "0x02DEd5a7EDDA750E3Eb240b54437a54d57b74dBE",   # PEPE
    138: "0xdc49F292ad1bb3DAb6C11363d74ED06F38b9bd9C",   # APT
    141: "0x205aaD468a11fd5D34fA7211bC6Bad5b3deB9b98",   # OP
    144: "0x4096b9bfB4c34497B7a3939D4f629cf65EBf5634",   # TIA
    145: "0x3a9659C071dD3C37a8b1A2363409A8D41B2Feae3",   # STX
    153: "0x4a85B128EBDaFC24d5CB611e161376ffDECeB289",   # SUI
    159: "0xCc9742d77622eE9abBF1Df03530594f9097bDcB3",   # SEI
    167: "0x4bC735Ef24bf286983024CAd5D03f0738865Aaef",   # 1INCH
    187: "0x594b919AD828e693B935705c3F816221729E7AE8",   # WTI
    194: "0x66853E19d73c0F9301fe099c324A1E9726953433",   # PENDLE
    205: "0xF7Ee427318d2Bd0EEd3c63382D0d52Ad8A68f90D",   # WIF
    216: "0x37DDEE84dE03d039e1Bf809b7a01EDd2c4665771",   # MNT
    219: "0x9eE96caa9972c801058CAA8E23419fc6516FbF7e",   # ENA
    223: "0x6aCcBB82aF71B8a576B4C05D4aF92A83A035B991",   # TAO
    236: "0x1940fEd49cDBC397941f2D336eb4994D599e568B",   # ZRO
    269: "0x82BA56a2fADF9C14f17D08bc51bDA0bDB83A8934",   # POL
    328: "0x373510BDa1ab7e873c731968f4D81B685f520E4B",   # TRUMP
    329: "0xE2CB592D636c500a6e469628054F09d58e4d91BB",   # MELANIA
    331: "0xf9ce4fE2F0EcE0362cb416844AE179a49591D567",   # HYPE
    376: "0x950DC95D4E537A14283059bADC2734977C454498",   # COIN
    407: "0x0C997958ccE7A0403AEA7E34d14bbaDA897B5bb3",   # PUMP
    413: "0x4b13Dd76De990Db9A2Dab58D35C2c02E5e3AE848",   # WLFI
    416: "0xea320E4d688B143A3bFBF1b4a5cc4B986fCa086c",   # CRO
    418: "0x1b47b4124b9A5094C59710E6b9126e5e32a4fb8E",   # XPL
    433: "0x0225781042C46dB247e009FFEAd5aEf044f3E7BE",   # MON
}  # fmt: skip

# 从 Gains 官方 pair list 确定所有 pairIndex
# 来源: https://docs.gains.trade/gtrade-leveraged-trading/pair-list
# 包含所有 Active 的加密/股票/指数/外汇/大宗商品
_PAIR_INDICES: dict[str, int] = {
    # Crypto
    "BTC": 0,
    "ETH": 1,
    "LINK": 2,
    "DOGE": 3,
    "ADA": 5,
    "AAVE": 7,
    "ALGO": 8,
    "BAT": 9,
    "COMP": 10,
    "DOT": 11,
    "MANA": 14,
    "UNI": 17,
    "XLM": 18,
    "XRP": 19,
    "ZEC": 20,
    "SOL": 33,
    "XTZ": 34,
    "BCH": 35,
    "CRV": 37,
    "DASH": 38,
    "ETC": 39,
    "ICP": 40,
    "THETA": 43,
    "TRX": 44,
    "SAND": 46,
    "BNB": 47,
    "GRT": 49,
    "HBAR": 50,
    "XMR": 51,
    "CHZ": 56,
    "SHIB": 57,
    "AVAX": 102,
    "ATOM": 103,
    "NEAR": 104,
    "QNT": 105,
    "TON": 107,
    "ARB": 109,
    "LDO": 128,
    "INJ": 129,
    "CAKE": 131,
    "TWT": 133,
    "PEPE": 134,
    "FIL": 137,
    "APT": 138,
    "IMX": 139,
    "VET": 140,
    "OP": 141,
    "RNDR": 142,
    "EGLD": 143,
    "TIA": 144,
    "STX": 145,
    "GALA": 148,
    "SUI": 153,
    "FET": 155,
    "CFX": 156,
    "AR": 158,
    "SEI": 159,
    "1INCH": 167,
    "FLOKI": 168,
    "WLD": 171,
    "ENS": 175,
    "JUP": 191,
    "BONK": 193,
    "PENDLE": 194,
    "STRK": 200,
    "PYTH": 203,
    "WIF": 205,
    "ETHFI": 212,
    "ONDO": 215,
    "MNT": 216,
    "KAS": 217,
    "ENA": 219,
    "TAO": 223,
    "ZRO": 236,
    "ZK": 237,
    "JASMY": 249,
    "SUN": 264,
    "POL": 269,
    "EIGEN": 282,
    "AERO": 286,
    "BSV": 293,
    "RAY": 299,
    "BTCDEGEN": 300,
    "ZEN": 304,
    "VIRTUAL": 307,
    "ETHDEGEN": 313,
    "SOLDEGEN": 314,
    "PENGU": 320,
    "FARTCOIN": 321,
    "BNBDEGEN": 327,
    "TRUMP": 328,
    "MELANIA": 329,
    "HYPE": 331,
    "S": 332,
    "VVV": 339,
    "BANANAS31": 363,
    "SYRUP": 373,
    "LPT": 383,
    "BVIV": 384,
    "EVIV": 385,
    "B": 391,
    "H": 404,
    "PUMP": 407,
    "WLFI": 413,
    "ASTER": 414,
    "OKB": 415,
    "CRO": 416,
    "SKY": 417,
    "XPL": 418,
    "FLUID": 425,
    "MON": 433,
    "LIT": 442,
    "DCR": 445,
    "HYPEDEGEN": 452,
    "MEGA": 453,
    # Stocks (Active)
    "AAPL": 58,
    "MSFT": 62,
    "SNAP": 64,
    "NVDA": 65,
    "V": 66,
    "MA": 67,
    "PFE": 68,
    "KO": 69,
    "DIS": 70,
    "NKE": 72,
    "AMD": 73,
    "PYPL": 74,
    "ABNB": 75,
    "BA": 76,
    "SBUX": 77,
    "WMT": 78,
    "INTC": 79,
    "MCD": 80,
    "META": 81,
    "GOOGL": 82,
    "GME": 83,
    "AMZN": 84,
    "TSLA": 85,
    "BIDU": 395,
    "ROKU": 396,
    "LMT": 397,
    "RIOT": 398,
    "MARA": 399,
    "COIN": 376,
    "HOOD": 377,
    "MSTR": 378,
    "CRCL": 386,
    "SBET": 393,
    "PLTR": 394,
    "NFLX": 439,
    "WPM": 448,
    # Indices (Active)
    "SPY": 86,
    "QQQ": 87,
    "IWM": 88,
    "DIA": 89,
    "SPX500": 436,
    "NAS100": 437,
    "USA30": 438,
    "GDX": 446,
    "URA": 447,
    "URNM": 451,
    # Forex - Major (Active)
    "EUR": 21,
    "JPY": 22,
    "GBP": 23,
    "CAD": 26,
    # Commodities (Active)
    "XAU": 90,
    "XAG": 91,
    "WTI": 187,
    "XPT": 188,
    "XPD": 189,
    "HG": 190,
    "NATGAS": 449,
    "BRENT": 450,
}

# pairIndex → 资产名 反向查找表（由 _PAIR_INDICES 自动生成）
_PAIR_SYMBOLS: dict[int, str] = {v: k for k, v in _PAIR_INDICES.items()}

# 最小 ABI：仅包含 adapter 需要的函数和事件
_DIAMOND_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "address", "name": "user", "type": "address"},
                    {"internalType": "uint32", "name": "index", "type": "uint32"},
                    {"internalType": "uint16", "name": "pairIndex", "type": "uint16"},
                    {"internalType": "uint24", "name": "leverage", "type": "uint24"},
                    {"internalType": "bool", "name": "long", "type": "bool"},
                    {"internalType": "bool", "name": "isOpen", "type": "bool"},
                    {"internalType": "uint8", "name": "collateralIndex", "type": "uint8"},
                    {"internalType": "uint8", "name": "tradeType", "type": "uint8"},
                    {"internalType": "uint120", "name": "collateralAmount", "type": "uint120"},
                    {"internalType": "uint64", "name": "openPrice", "type": "uint64"},
                    {"internalType": "uint64", "name": "tp", "type": "uint64"},
                    {"internalType": "uint64", "name": "sl", "type": "uint64"},
                    {"internalType": "bool", "name": "isCounterTrade", "type": "bool"},
                    {"internalType": "uint160", "name": "positionSizeToken", "type": "uint160"},
                    {"internalType": "uint24", "name": "__placeholder", "type": "uint24"},
                ],
                "internalType": "struct ITradingStorage.Trade",
                "name": "_trade",
                "type": "tuple",
            },
            {"internalType": "uint16", "name": "_maxSlippageP", "type": "uint16"},
            {"internalType": "address", "name": "_referrer", "type": "address"},
        ],
        "name": "openTrade",
        "outputs": [{"internalType": "uint32", "name": "_index", "type": "uint32"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint32", "name": "_index", "type": "uint32"},
            {"internalType": "uint64", "name": "_expectedPrice", "type": "uint64"},
        ],
        "name": "closeTradeMarket",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "_trader", "type": "address"}],
        "name": "getTrades",
        "outputs": [
            {
                "components": [
                    {"internalType": "address", "name": "user", "type": "address"},
                    {"internalType": "uint32", "name": "index", "type": "uint32"},
                    {"internalType": "uint16", "name": "pairIndex", "type": "uint16"},
                    {"internalType": "uint24", "name": "leverage", "type": "uint24"},
                    {"internalType": "bool", "name": "long", "type": "bool"},
                    {"internalType": "bool", "name": "isOpen", "type": "bool"},
                    {"internalType": "uint8", "name": "collateralIndex", "type": "uint8"},
                    {"internalType": "uint8", "name": "tradeType", "type": "uint8"},
                    {"internalType": "uint120", "name": "collateralAmount", "type": "uint120"},
                    {"internalType": "uint64", "name": "openPrice", "type": "uint64"},
                    {"internalType": "uint64", "name": "tp", "type": "uint64"},
                    {"internalType": "uint64", "name": "sl", "type": "uint64"},
                    {"internalType": "bool", "name": "isCounterTrade", "type": "bool"},
                    {"internalType": "uint160", "name": "positionSizeToken", "type": "uint160"},
                    {"internalType": "uint24", "name": "__placeholder", "type": "uint24"},
                ],
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getCollaterals",
        "outputs": [
            {
                "components": [
                    {"internalType": "address", "name": "collateral", "type": "address"},
                    {"internalType": "bool", "name": "isActive", "type": "bool"},
                    {"internalType": "uint88", "name": "__placeholder", "type": "uint88"},
                    {"internalType": "uint128", "name": "precision", "type": "uint128"},
                    {"internalType": "uint128", "name": "precisionDelta", "type": "uint128"},
                ],
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "address", "name": "trader", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "pairIndex", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "index", "type": "uint256"},
            {"indexed": False, "internalType": "bool", "name": "isLong", "type": "bool"},
            {"indexed": False, "internalType": "uint256", "name": "collateralAmount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "leverage", "type": "uint256"},
        ],
        "name": "MarketOrderInitiated",
        "type": "event",
    },
]

# USDC ERC20 approve 和 allowance ABI（最小化）
_ERC20_APPROVE_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "remaining", "type": "uint256"}],
        "type": "function",
    },
]


def _get_pair_index(symbol: str) -> int:
    """通过资产名获取 Gains pairIndex（BTC -> 0, ETH -> 1）。"""
    upper = symbol.upper().split("/")[0]
    idx = _PAIR_INDICES.get(upper)
    if idx is not None:
        return idx
    raise ValueError(
        f"未知 pair: {symbol}（已注册: {list(_PAIR_INDICES)}），请手动添加或使用 --pair-index 指定"
    )


# ── Binance 价格映射 ──
# Gains symbol → Binance symbol (USDT 永续合约现货价格)
_BINANCE_SYMBOL_MAP: dict[str, str] = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT", "SOL": "SOLUSDT", "AVAX": "AVAXUSDT",
    "ATOM": "ATOMUSDT", "DOT": "DOTUSDT", "UNI": "UNIUSDT",
    "MATIC": "POLUSDT", "ARB": "ARBUSDT", "AAVE": "AAVEUSDT",
    "LTC": "LTCUSDT", "ADA": "ADAUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "CRV": "CRVUSDT", "GRT": "GRTUSDT",
    "NEAR": "NEARUSDT", "OP": "OPUSDT", "APT": "APTUSDT",
    "SUI": "SUIUSDT", "SEI": "SEIUSDT", "TIA": "TIAUSDT",
    "TON": "TONUSDT", "INJ": "INJUSDT", "STX": "STXUSDT",
    "FIL": "FILUSDT", "ICP": "ICPUSDT", "RUNE": "RUNEUSDT",
    "PEPE": "PEPEUSDT", "SHIB": "SHIBUSDT",
    # 商品/指数 — Binance 没有的留空，会 fallback
}

# Binance API 缓存
_binance_cache: dict[str, tuple[float, float]] = {}  # symbol → (price, timestamp)
_BINANCE_CACHE_TTL = 30  # 秒


def _get_yahoo_v8_price(symbol: str) -> float:
    """通过 Yahoo Finance v8 REST API 获取股票/ETF 价格。
    比 yfinance 库更轻量，不易触发限流。返回 0 表示失败。"""
    # 修正一些 ticker 格式
    yahoo_symbol = symbol.upper().replace("^", "")
    if yahoo_symbol in ("SPX500",):
        yahoo_symbol = "^GSPC"
    elif yahoo_symbol in ("NAS100",):
        yahoo_symbol = "^NDX"
    elif yahoo_symbol in ("USA30",):
        yahoo_symbol = "^DJI"
    elif yahoo_symbol in ("WTI",):
        yahoo_symbol = "CL=F"
    elif yahoo_symbol in ("BRENT",):
        yahoo_symbol = "BZ=F"
    elif yahoo_symbol in ("XAU",):
        yahoo_symbol = "GC=F"
    elif yahoo_symbol in ("XAG",):
        yahoo_symbol = "SI=F"
    elif yahoo_symbol in ("NATGAS",):
        yahoo_symbol = "NG=F"
    elif yahoo_symbol in ("XPT",):
        yahoo_symbol = "PL=F"
    elif yahoo_symbol in ("XPD",):
        yahoo_symbol = "PA=F"
    elif yahoo_symbol in ("HG",):
        yahoo_symbol = "HG=F"

    try:
        import requests as _req
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        resp = _req.get(url, timeout=10, headers=headers)
        if resp.status_code != 200:
            return 0.0
        data = resp.json()
        result = data["chart"]["result"][0]
        price = result["meta"]["regularMarketPrice"]
        return float(price)
    except Exception:
        return 0.0


def _get_binance_price(symbol: str) -> float:
    """通过 Binance REST API 获取现货价格（国内可访问）。返回 0 表示失败。"""
    bn_symbol = _BINANCE_SYMBOL_MAP.get(symbol.upper())
    if bn_symbol is None:
        return 0.0

    now = __import__('time').time()
    if bn_symbol in _binance_cache:
        cached_price, cached_ts = _binance_cache[bn_symbol]
        if now - cached_ts < _BINANCE_CACHE_TTL:
            return cached_price

    try:
        import requests as _req
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={bn_symbol}"
        resp = _req.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            price = float(data["price"])
            _binance_cache[bn_symbol] = (price, now)
            return price
    except Exception:
        pass
    return 0.0


def _decode_trades(raw_trades: list[Any]) -> list[dict]:
    """将合约返回的 tuple[] 转为可读的 dict 列表。"""
    results = []
    for t in raw_trades:
        if not t[5]:  # isOpen == False
            continue
        results.append({
            "index": t[1],
            "pair_index": t[2],
            "leverage": t[3] / 1000,
            "long": t[4],
            "collateral_index": t[6],
            "collateral_amount_raw": t[7],
            "open_price_raw": t[8],
            "tp_raw": t[9],
            "sl_raw": t[10],
        })
    return results


class GainsVenueAdapter(OnchainVenueAdapter):
    def __init__(self, config: OnchainConfig):
        super().__init__(config)
        self._web3: Optional["Web3"] = None
        self._contract: Any = None
        self._usdc_collateral_index: int | None = None
        self._usdc_contract_address: str | None = None
        self._price_provider: Any = None  # GtradeDataProvider（WebSocket 价格源）

    def set_price_provider(self, provider: Any):
        """注入 gTrade WebSocket 价格源（GtradeDataProvider 实例）。

        注入后，_get_current_price 优先使用 WebSocket 实时 mark 价格，
        Chainlink / REST / yfinance 作为降级路径。
        """
        self._price_provider = provider

    def _get_web3(self) -> "Web3":
        if self._web3 is not None:
            return self._web3
        if not self.config.rpc_url:
            raise ValueError("onchain 非 dry_run 模式需要配置 rpc_url")
        try:
            from web3 import Web3
        except ImportError as e:
            raise ImportError("请安装 onchain 依赖: uv sync --extra onchain") from e

        self._web3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        if not self._web3.is_connected():
            raise ConnectionError(f"无法连接 RPC: {self.config.rpc_url}")
        self._contract = self._web3.eth.contract(address=DIAMOND_ADDRESS, abi=_DIAMOND_ABI)
        return self._web3

    def _get_usdc_collateral_info(self) -> tuple[int, str]:
        """返回 (usdc_collateral_index, usdc_contract_address)，带缓存。"""
        if self._usdc_collateral_index is not None and self._usdc_contract_address is not None:
            return self._usdc_collateral_index, self._usdc_contract_address
        w3 = self._get_web3()
        contract = w3.eth.contract(address=DIAMOND_ADDRESS, abi=_DIAMOND_ABI)
        collaterals = contract.functions.getCollaterals().call()
        for i, col in enumerate(collaterals):
            addr = col[0]
            precision = col[3]
            if precision == 1_000_000:
                self._usdc_collateral_index = i
                self._usdc_contract_address = addr
                _logger.info(f"USDC collateral index: {i}, address: {addr}")
                return i, addr
        raise ValueError("未在 Gains 合约中找到 USDC collateral (精度=6位小数)")

    def ensure_collateral_allowance(self, required_amount_usdc: float) -> bool:
        """检查并自动授权 USDC 给 Gains Diamond 合约。

        返回 True 表示已有足够额度或授权成功，False 表示额度不足（dry-run 模式）。
        在实盘模式下会自动发送 approve 交易。
        """
        w3 = self._get_web3()
        sender = w3.to_checksum_address(self.config.wallet_address)
        _, usdc_addr = self._get_usdc_collateral_info()
        erc20 = w3.eth.contract(
            address=w3.to_checksum_address(usdc_addr),
            abi=_ERC20_APPROVE_ABI,
        )
        diamond = w3.to_checksum_address(DIAMOND_ADDRESS)

        required_raw = int(required_amount_usdc * 10**6)
        current_allowance = erc20.functions.allowance(sender, diamond).call()

        if current_allowance >= required_raw:
            _logger.info(
                f"USDC 额度充足: {current_allowance / 10**6:.2f} >= {required_amount_usdc:.2f}"
            )
            return True

        if self.config.dry_run:
            _logger.warning(
                f"USDC 额度不足: 当前 {current_allowance / 10**6:.2f}, "
                f"需要 {required_amount_usdc:.2f} "
                f"(dry-run 模式, 不会自动 approve)"
            )
            return False

        _logger.info(
            f"USDC 额度不足: 当前 {current_allowance / 10**6:.2f}, "
            f"需要 {required_amount_usdc:.2f}, "
            f"正在发送 approve 交易..."
        )

        tx = erc20.functions.approve(diamond, required_raw).build_transaction({
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender),
            "gas": 100_000,
            **self._build_gas_params(w3),
        })
        signed = w3.eth.account.sign_transaction(tx, private_key=self.config.private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        _logger.info(f"USDC approve tx sent: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt["status"] == 0:
            raise RuntimeError(f"USDC approve 交易失败: {tx_hash.hex()}")
        _logger.info(f"USDC approve 成功: {tx_hash.hex()}")
        return True

    def _build_gas_params(self, w3: "Web3") -> dict:
        """构建 EIP-1559 gas 参数，对 baseFee 加 30% 缓冲避免瞬变拒绝。"""
        latest = w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas")
        if base_fee is None:
            return {"gasPrice": w3.eth.gas_price}
        try:
            priority = max(w3.eth.max_priority_fee or 0, w3.to_wei(MIN_PRIORITY_FEE_GWEI, "gwei"))
        except Exception:
            priority = base_fee // 10
        max_fee = int(base_fee * 1.3) + priority
        return {
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority,
        }

    def _get_current_price(self, pair_index: int) -> int:
        """获取当前价格，返回 Gains openPrice 格式（1e10 精度）。

        价格源优先级（从快到准）：
        1. gTrade WebSocket v4 实时 mark 价格（覆盖全部品种）
        2. Chainlink 预言机（链上权威，仅部分品种有 feed）
        3. gTrade charts REST API
        4. yfinance 现货
        """
        # ──  Tier 1: gTrade WebSocket mark 价格 ──
        if self._price_provider is not None:
            try:
                ws_price = self._price_provider.get_price(pair_index)
                if ws_price is not None and ws_price > 0:
                    open_price = int(ws_price * 10**10)
                    _logger.debug(
                        f"gTrade WS price pairIndex={pair_index}: {ws_price:.4f} USD "
                        f"(openPrice={open_price})"
                    )
                    return open_price
            except Exception as exc:
                _logger.warning(f"gTrade WebSocket 价格查询失败: {exc}")

        # ──  Tier 2: Chainlink 预言机 ──
        feed_address = _CHAINLINK_FEEDS.get(pair_index)
        if feed_address is not None:
            w3 = self._get_web3()
            feed = w3.eth.contract(
                address=w3.to_checksum_address(feed_address),
                abi=_CHAINLINK_ABI,
            )
            try:
                _, answer, _, updated_at, _ = feed.functions.latestRoundData().call()
                feed_decimals = feed.functions.decimals().call()
                scale = 10 ** (10 - feed_decimals)
                open_price = int(answer * scale)
                age_seconds = int(w3.eth.get_block("latest")["timestamp"] - updated_at)
                if age_seconds > 300:
                    _logger.warning(
                        f"Chainlink feed 价格已 {age_seconds}s 未更新 (pairIndex={pair_index})"
                    )
                _logger.info(
                    f"Chainlink price pairIndex={pair_index}: "
                    f"{answer / 10**feed_decimals:.4f} USD "
                    f"(openPrice={open_price})"
                )
                return open_price
            except Exception as exc:
                _logger.warning(f"查询 Chainlink feed {feed_address} 失败: {exc}")

        # ──  Tier 3: Yahoo Finance v8 REST API（轻量，不易限流）──
        symbol = _PAIR_SYMBOLS.get(pair_index)
        if symbol:
            try:
                yh_price = _get_yahoo_v8_price(symbol)
                if yh_price > 0:
                    open_price = int(yh_price * 10**10)
                    _logger.info(f"Yahoo v8 price pairIndex={pair_index} {symbol}: {yh_price:.4f} USD (openPrice={open_price})")
                    return open_price
            except Exception as exc:
                _logger.debug(f"Yahoo v8 获取 {symbol} 价格失败: {exc}")

        # ──  Tier 4: Binance REST API（国内可访问）──
        symbol = _PAIR_SYMBOLS.get(pair_index)
        if symbol:
            try:
                bn_price = _get_binance_price(symbol)
                if bn_price > 0:
                    open_price = int(bn_price * 10**10)
                    _logger.info(f"Binance price pairIndex={pair_index} {symbol}: {bn_price:.4f} USD (openPrice={open_price})")
                    return open_price
            except Exception as exc:
                _logger.debug(f"Binance 获取 {symbol} 价格失败: {exc}")

        # ──  Tier 4: gTrade charts REST API ──
        try:
            now_ts = int(__import__('time').time())
            from_ts = now_ts - 86400
            charts_url = f"https://backend-arbitrum.gains.trade/charts?pairIndex={pair_index}&from={from_ts}&to={now_ts}&resolution=5"
            resp = __import__('requests').get(charts_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    price = float(data[-1].get('close', 0))
                    if price > 0:
                        open_price = int(price * 10**10)
                        _logger.info(f"gTrade charts price pairIndex={pair_index}: {price:.4f} USD (openPrice={open_price})")
                        return open_price
        except Exception as exc:
            _logger.warning(f"gTrade charts 获取 pairIndex={pair_index} 价格失败: {exc}")

        # ──  Tier 4: yfinance 现货 ──
        symbol = _PAIR_SYMBOLS.get(pair_index)
        if symbol:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    if price > 0:
                        open_price = int(price * 10**10)
                        _logger.info(f"yfinance price pairIndex={pair_index} {symbol}: {price:.4f} USD (openPrice={open_price})")
                        return open_price
            except Exception as exc:
                _logger.warning(f"yfinance 获取 {symbol} 价格失败: {exc}")

        _logger.warning(f"pairIndex={pair_index} 无可用价格源，openPrice 设为 0（合约可能拒绝）")
        return 0

    def _ensure_live_config(self):
        if self.config.dry_run:
            return
        if not self.config.wallet_address or self.config.wallet_address == ZERO_ADDRESS:
            raise ValueError("onchain 实盘需要配置 wallet_address")
        if not self.config.private_key or self.config.private_key == "0x" + "0" * 64:
            raise ValueError("onchain 实盘需要配置 private_key")

    def simulate_open(self, symbol: str, side: str, collateral: float, leverage: float) -> dict:
        """预检开仓：用 eth_call 模拟交易，不花 gas。返回 {'ok': bool, 'reason': str}"""
        try:
            self._ensure_live_config()
            w3 = self._get_web3()
            contract = self._contract
            sender = w3.to_checksum_address(self.config.wallet_address)

            pair_index = _get_pair_index(symbol)
            collateral_index = self._get_usdc_collateral_info()[0] + 1
            collateral_amount = int(collateral * 10**6)
            max_slippage_p = int(0.01 * 100_000)
            open_price = self._get_current_price(pair_index)
            if open_price == 0:
                return {'ok': False, 'reason': '无价格源，openPrice=0'}

            trade_tuple = (
                sender, 0, pair_index, int(leverage * 1000),
                side == "long", True, collateral_index, 0,
                collateral_amount, open_price, 0, 0,
                False, 0, 0,
            )

            contract.functions.openTrade(
                trade_tuple, max_slippage_p, ZERO_ADDRESS
            ).call({
                "from": sender,
                "gas": GAS_LIMIT_OPEN_TRADE,
            })
            return {'ok': True, 'reason': ''}
        except Exception as e:
            err = str(e)
            # 提取有用的错误信息
            if 'InsufficientCollateral' in err or '3a23d825' in err:
                return {'ok': False, 'reason': '保证金不足'}
            if '0xc5723b51' in err:
                return {'ok': False, 'reason': '品种/函数不存在'}
            # 截取前150字符作为reason
            reason = err.split("'")[1] if "'" in err else err[:100]
            return {'ok': False, 'reason': reason}

    def open_trade(self, req: OnchainOpenRequest) -> OnchainTradeResult:
        order_id = f"gains-open-{uuid.uuid4().hex[:12]}"
        if self.config.dry_run:
            _logger.info(f"[dry_run] open_trade {req}")
            return OnchainTradeResult(
                tx_hash=None,
                order_sys_id=order_id,
                symbol=req.symbol,
                side=req.side,
                status="filled",
                raw={"dry_run": True, "request": req.__dict__},
            )

        self._ensure_live_config()
        w3 = self._get_web3()
        contract = self._contract
        sender = w3.to_checksum_address(self.config.wallet_address)

        # 先确保 USDC 授权足够
        self.ensure_collateral_allowance(req.collateral)

        pair_index = _get_pair_index(req.symbol)
        # Gains v10 Trade struct 中 collateralIndex 是 1-based
        collateral_index = self._get_usdc_collateral_info()[0] + 1
        # USDC 6 位小数 -> wei 单位
        collateral_amount = int(req.collateral * 10**6)
        # req.slippage 是百分比小数: 0.01 = 1%，合约用 1e3 精度 (1000 = 1%)
        max_slippage_p = int(req.slippage * 100_000)

        # 获取当前市价作为 openPrice（市价单的滑点校准参考）
        open_price = self._get_current_price(pair_index)

        trade_tuple = (
            sender,           # user
            0,                # index (0 for new trade)
            pair_index,       # pairIndex
            int(req.leverage * 1000),  # leverage (1e3 precision)
            req.side == "long",  # long
            True,             # isOpen
            collateral_index, # collateralIndex
            0,                # tradeType (0=TRADE, 市价单)
            collateral_amount,  # collateralAmount
            open_price,       # openPrice (当前市价，合约用做滑点校准)
            0,                # tp
            0,                # sl
            False,            # isCounterTrade
            0,                # positionSizeToken (0 = computed by contract)
            0,                # __placeholder
        )

        tx = contract.functions.openTrade(
            trade_tuple, max_slippage_p, ZERO_ADDRESS
        ).build_transaction({
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender),
            "gas": GAS_LIMIT_OPEN_TRADE,
            **self._build_gas_params(w3),
        })

        signed = w3.eth.account.sign_transaction(tx, private_key=self.config.private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        _logger.info(f"open_trade tx sent: {tx_hash.hex()}")
        receipt: TxReceipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] == 0:
            _logger.error(f"发送的请求参数: {receipt.__dict__}")
            raise RuntimeError(f"open_trade 交易失败: {tx_hash.hex()}")

        # 从事件日志或返回值获取 trade index
        logs = contract.events.MarketOrderInitiated().process_receipt(receipt)
        trade_index = logs[0]["args"]["index"] if logs else 0

        return OnchainTradeResult(
            tx_hash=tx_hash.hex(),
            order_sys_id=str(trade_index),
            symbol=req.symbol,
            side=req.side,
            status="filled",
            raw={"tx_hash": tx_hash.hex(), "trade_index": trade_index, "receipt_status": receipt["status"]},
        )

    def close_trade(self, req: OnchainCloseRequest) -> OnchainTradeResult:
        order_id = f"gains-close-{uuid.uuid4().hex[:12]}"
        if self.config.dry_run:
            _logger.info(f"[dry_run] close_trade {req}")
            return OnchainTradeResult(
                tx_hash=None,
                order_sys_id=order_id,
                symbol=req.symbol,
                side="close",
                status="filled",
                raw={"dry_run": True, "request": req.__dict__},
            )

        self._ensure_live_config()
        w3 = self._get_web3()
        contract = self._contract
        sender = w3.to_checksum_address(self.config.wallet_address)

        # 获取用户的持仓以找到要平的仓位
        positions = self.fetch_positions()

        if req.position_id:
            # 指定具体仓位 index
            matching = [p for p in positions if str(p["index"]) == req.position_id]
            if not matching:
                raise ValueError(f"没有找到 position_id={req.position_id} 的持仓")
            position = matching[0]
            # 从 pair_index 解析实际 symbol（可能和 req.symbol 不同）
            resolved_symbol = _PAIR_SYMBOLS.get(position["pair_index"], req.symbol)
        else:
            # 按 symbol 找第一个仓位
            target_positions = [
                p for p in positions
                if p["pair_index"] == _get_pair_index(req.symbol)
            ]
            if not target_positions:
                raise ValueError(f"没有找到 {req.symbol} 的持仓")
            position = target_positions[0]
            resolved_symbol = req.symbol
        trade_index = position["index"]
        is_long = position["long"]
        pair_index = position["pair_index"]

        # 获取当前市价作为 expectedPrice（平仓滑点校准）
        current_price = self._get_current_price(pair_index)
        slippage_p = int(req.slippage * 100_000)  # 1e3 精度，1000 = 1%
        if current_price > 0:
            if is_long:
                # 多头平仓（卖出）：expectedPrice 是接受的最低价格 = 市价×(1 - slippage)
                expected_price = int(current_price * (1_000_000 - slippage_p) / 1_000_000)
            else:
                # 空头平仓（买入）：expectedPrice 是接受的最高价格 = 市价×(1 + slippage)
                expected_price = int(current_price * (1_000_000 + slippage_p) / 1_000_000)
        else:
            expected_price = 0

        _logger.info(
            f"close_trade index={trade_index} pair={pair_index} "
            f"is_long={is_long} current_price={current_price} "
            f"expected_price={expected_price} slippage={req.slippage*100:.1f}%"
        )

        tx = contract.functions.closeTradeMarket(trade_index, expected_price).build_transaction({
            "from": sender,
            "nonce": w3.eth.get_transaction_count(sender),
            "gas": GAS_LIMIT_CLOSE_TRADE,
            **self._build_gas_params(w3),
        })

        signed = w3.eth.account.sign_transaction(tx, private_key=self.config.private_key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        _logger.info(f"close_trade tx sent: {tx_hash.hex()}")
        receipt: TxReceipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] == 0:
            raise RuntimeError(f"close_trade 交易失败: {tx_hash.hex()}")

        return OnchainTradeResult(
            tx_hash=tx_hash.hex(),
            order_sys_id=order_id,
            symbol=resolved_symbol,
            side="close",
            status="filled",
            raw={"tx_hash": tx_hash.hex(), "trade_index": trade_index, "receipt_status": receipt["status"]},
        )

    def fetch_positions(self) -> list[dict]:
        if self.config.dry_run:
            return []
        self._ensure_live_config()
        w3 = self._get_web3()
        contract = self._contract
        sender = w3.to_checksum_address(self.config.wallet_address)
        raw = contract.functions.getTrades(sender).call()
        return _decode_trades(raw)

    def fetch_wallet_balance(self, token: str) -> float:
        if self.config.dry_run:
            return 0.0
        w3 = self._get_web3()
        if token.upper() in ("ETH", "NATIVE"):
            checksum_addr = w3.to_checksum_address(self.config.wallet_address)
            bal = w3.eth.get_balance(checksum_addr)
            return float(w3.from_wei(bal, "ether"))
        raise NotImplementedError(f"ERC20 余额查询待实现: {token}")
