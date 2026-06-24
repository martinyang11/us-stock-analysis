"""Canonical gTrade TradFi universe for SA/SANN/CatTrader.

The order below defines the stable SANN variety_id mapping.
"""

STOCK_SYMBOLS = [
    "NVDA",
    "AAPL",
    "AMZN",
    "COIN",
    "CRCL",
    "GME",
    "GOOGL",
    "HOOD",
    "LMT",
    "MARA",
    "MCD",
    "META",
    "MSFT",
    "MSTR",
    "NFLX",
    "PLTR",
    "PYPL",
    "RIOT",
    "SNAP",
    "SPCX",
    "TSLA",
    "WPM",
]

INDEX_SYMBOLS = [
    "SPY",
    "DIA",
    "GDX",
    "IWM",
    "QQQ",
    "URA",
    "URNM",
]

TRADFI_SYMBOLS = STOCK_SYMBOLS + INDEX_SYMBOLS
TRADFI_SYMBOL_SET = set(TRADFI_SYMBOLS)
NUM_TRADFI_SYMBOLS = len(TRADFI_SYMBOLS)

# gTrade has renamed/aliased a few pairs over time. Keep the canonical symbol
# matching the UI screenshots while accepting older API/source names.
GTRADE_NAME_ALIASES = {
    "SPX500": "SPCX",
}

YF_TICKER_OVERRIDES = {
    "SPCX": "^GSPC",
    "SPX500": "^GSPC",
}

CATEGORY_OVERRIDES = {
    "SPCX": "ETF/指数",
}

GTRADE_PAIR_INDICES = {
    "NVDA": 65,
    "AAPL": 58,
    "AMZN": 84,
    "COIN": 376,
    "CRCL": 386,
    "GME": 83,
    "GOOGL": 82,
    "HOOD": 377,
    "LMT": 397,
    "MARA": 399,
    "MCD": 80,
    "META": 81,
    "MSFT": 62,
    "MSTR": 378,
    "NFLX": 439,
    "PLTR": 394,
    "PYPL": 74,
    "RIOT": 398,
    "SNAP": 64,
    "SPCX": 436,
    "TSLA": 85,
    "WPM": 448,
    "SPY": 86,
    "DIA": 89,
    "GDX": 446,
    "IWM": 88,
    "QQQ": 87,
    "URA": 447,
    "URNM": 451,
}
