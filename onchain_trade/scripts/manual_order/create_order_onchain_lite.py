"""
链上（gTrade）开平仓 Lite 脚本：直接通过 web3 调用 Gains 合约。
不依赖数据库和 InterMTE。

支持 Crypto / Stocks / Indices / Forex / Commodities 全天候交易。
注意：美股/指数在非交易时段(周末/盘后)可能无法开仓，加密 7x24 可用。

用法示例:
  # 开 BTC 多头（dry-run）
  uv run python scripts/manual_order/create_order_onchain_lite.py \\
    --symbol BTC --side long --collateral 10 --leverage 2

  # 开 AAPL 多头（dry-run，需美股交易时段）
  uv run python scripts/manual_order/create_order_onchain_lite.py \\
    --symbol AAPL --side long --collateral 50 --leverage 5

  # 开 ETH 空头（dry-run）
  uv run python scripts/manual_order/create_order_onchain_lite.py \\
    --symbol ETH --side short --collateral 20 --leverage 3

  # 平 BTC 仓位
  uv run python scripts/manual_order/create_order_onchain_lite.py \\
    --symbol BTC --close

常用 pairIndex（完整列表见 adapter.py）:
  Crypto: BTC(0), ETH(1), SOL(33), ARB(109)
  Stocks: AAPL(58), MSFT(62), NVDA(65), TSLA(85), META(81), SPY(86)
  Forex:  EUR/USD(21), USD/JPY(22), GBP/USD(23)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from hole_board.exchange.onchain.account_config import parse_onchain_config
from hole_board.exchange.onchain.venues.gains.adapter import (
    DIAMOND_ADDRESS,
    GAS_LIMIT_CLOSE_TRADE,
    GAS_LIMIT_OPEN_TRADE,
    GainsVenueAdapter,
    _PAIR_INDICES,
    _PAIR_SYMBOLS,
    _get_pair_index,
)
from hole_board.exchange.onchain.types import OnchainCloseRequest, OnchainOpenRequest

try:
    from dt_config.onchain_config import WALLET_ADDRESS, PRIVATE_KEY, ONCHAIN_DEFAULTS
except ImportError:
    WALLET_ADDRESS = PRIVATE_KEY = ""
    ONCHAIN_DEFAULTS = {}


def _print_gas_info(adapter: GainsVenueAdapter, is_close: bool) -> float:
    """查询 ETH 余额并估算 gas，返回 ETH 余额。"""
    w3 = adapter._get_web3()
    gas_params = adapter._build_gas_params(w3)
    gas_limit = GAS_LIMIT_CLOSE_TRADE if is_close else GAS_LIMIT_OPEN_TRADE
    eth_balance = adapter.fetch_wallet_balance("ETH")
    max_fee = gas_params.get("maxFeePerGas") or w3.eth.gas_price
    gas_cost_eth = gas_limit * max_fee / 10**18

    print(f"  ETH 余额: {eth_balance:.6f} ETH")
    print(f"  gas 预估: ~{gas_limit} unit × {max_fee/10**9:.2f} gwei = ~{gas_cost_eth:.6f} ETH")

    if eth_balance < gas_cost_eth:
        print(f"  ** ETH 余额不足！需要 ~{gas_cost_eth:.6f} ETH，当前仅 {eth_balance:.6f} ETH")

    return eth_balance


def _print_usdc_allowance(adapter: GainsVenueAdapter, required_usdc: float) -> None:
    """查询并打印 USDC 授权额度状态。"""
    w3 = adapter._get_web3()
    sender = w3.to_checksum_address(adapter.config.wallet_address)
    _, usdc_addr = adapter._get_usdc_collateral_info()
    diamond = w3.to_checksum_address(DIAMOND_ADDRESS)
    erc20_abi = [
        {"constant": True, "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "remaining", "type": "uint256"}], "type": "function"},
    ]
    erc20 = w3.eth.contract(address=w3.to_checksum_address(usdc_addr), abi=erc20_abi)
    allowance = erc20.functions.allowance(sender, diamond).call()
    allowance_usdc = allowance / 10**6
    print(f"  USDC 授权额度: {allowance_usdc:.2f} USDC (需要 {required_usdc:.2f})")
    if allowance_usdc < required_usdc:
        print(f"  ** USDC 授权不足！脚本会在实盘时自动追加授权")


def main() -> int:
    parser = argparse.ArgumentParser(description="链上合约（gTrade）Lite 开平仓，支持 Crypto / Stocks / Indices / Forex")
    parser.add_argument("--symbol", default="BTC", help="合约标的，如 BTC、ETH、SOL")
    parser.add_argument("--side", choices=["long", "short"], default="long", help="方向")
    parser.add_argument("--collateral", type=float, default=10, help="抵押品数量（USDC）")
    parser.add_argument("--leverage", type=float, default=1.1, help="杠杆倍数")
    parser.add_argument("--slippage", type=float, default=0.01, help="开仓滑点容忍度，默认 0.01 (1%)")
    parser.add_argument("--close-slippage", type=float, default=0.01, help="平仓滑点容忍度，默认 0.01 (1%)")
    parser.add_argument("--close", action="store_true", help="平仓模式（按 symbol 平掉第一个仓位）")
    parser.add_argument("--close-symbol", action="store_true", help="平仓指定 symbol 的全部仓位（配合 --symbol 使用）")
    parser.add_argument("--close-all", action="store_true", help="一键平掉所有持仓（逐个发交易）")
    parser.add_argument("--execute", action="store_true", help="实际发链上交易（不加则 dry-run）")

    args = parser.parse_args()

    # 检测合约是否已知
    pair_index = _PAIR_INDICES.get(args.symbol.upper())
    if pair_index is None:
        print(f"错误: 未知合约 {args.symbol}，已注册: {list(_PAIR_INDICES)}")
        return 1

    # 检测钱包配置
    if not WALLET_ADDRESS or WALLET_ADDRESS == "0x" + "0" * 40:
        print("错误: 请在 dt_config/onchain_config.py 中填写 WALLET_ADDRESS")
        return 1
    if not PRIVATE_KEY or PRIVATE_KEY == "0x" + "0" * 64:
        print("错误: 请在 dt_config/onchain_config.py 中填写 PRIVATE_KEY")
        return 1

    # 创建 adapter 实例（dry_run=False 才能正常查余额，实际交易由 lite 脚本控制）
    config = parse_onchain_config(
        api_name="okx_onchain",
        user_id=WALLET_ADDRESS,
        password=PRIVATE_KEY,
        onchain_venue=ONCHAIN_DEFAULTS.get("onchain_venue", "gains"),
        chain_id=ONCHAIN_DEFAULTS.get("chain_id", 42161),
        rpc_url=ONCHAIN_DEFAULTS.get("rpc_url", "https://arb1.arbitrum.io/rpc"),
        dry_run=False,
    )
    adapter = GainsVenueAdapter(config)

    if args.close:
        print(f"操作:   平仓 {args.symbol}  (close-slippage: {args.close_slippage*100:.1f}%)")
        if not args.execute:
            print(f"  dry-run：将查询 {WALLET_ADDRESS[:10]}... 的 {args.symbol} 持仓并发送平仓交易")
            print(f"\n  pairIndex: {pair_index}")
            print(f"  RPC:       {config.rpc_url}")
            _print_gas_info(adapter, is_close=True)
            print(f"\ndry-run 模式：加 --execute 后才会发送链上交易")
            return 0

        eth_balance = _print_gas_info(adapter, is_close=True)
        w3 = adapter._get_web3()
        gas_params = adapter._build_gas_params(w3)
        max_fee = gas_params.get("maxFeePerGas") or w3.eth.gas_price
        if eth_balance < GAS_LIMIT_CLOSE_TRADE * max_fee / 10**18:
            print("\n错误: ETH 余额不足以支付 gas，取消交易")
            return 1

        req = OnchainCloseRequest(symbol=args.symbol, slippage=args.close_slippage)
        result = adapter.close_trade(req)
        print(f"平仓成功: tx={result.tx_hash}")
        return 0

    if args.close_all:
        print(f"操作:   平仓所有持仓 (close-slippage: {args.close_slippage*100:.1f}%)")
        positions = adapter.fetch_positions()
        if not positions:
            print("  当前没有持仓")
            return 0

        print(f"  共 {len(positions)} 个持仓:")
        for p in positions:
            sym = _PAIR_SYMBOLS.get(p["pair_index"], f"pair:{p['pair_index']}")
            side = "LONG" if p["long"] else "SHORT"
            col = p["collateral_amount_raw"] / 10**6
            lev = p["leverage"]
            print(f"    index={p['index']}  {sym} {side}  {col:.2f} USDC  {lev:.1f}x")

        if not args.execute:
            print(f"\ndry-run 模式：加 --execute 后才会逐个发送平仓交易")
            return 0

        total = len(positions)
        for i, p in enumerate(positions, 1):
            sym = _PAIR_SYMBOLS.get(p["pair_index"], f"pair:{p['pair_index']}")
            print(f"\n[{i}/{total}] 平仓 {sym} index={p['index']}...", end=" ")
            req = OnchainCloseRequest(
                symbol=sym,
                position_id=str(p["index"]),
                slippage=args.close_slippage,
            )
            try:
                result = adapter.close_trade(req)
                print(f"✅ tx={result.tx_hash}")
            except Exception as e:
                print(f"❌ 失败: {e}")
                # 失败一个不中断，继续平其他
        return 0

    if args.close_symbol:
        pair_index = _PAIR_INDICES.get(args.symbol.upper())
        if pair_index is None:
            print(f"错误: 未知合约 {args.symbol}")
            return 1
        print(f"操作:   平仓 {args.symbol} 全部仓位 (close-slippage: {args.close_slippage*100:.1f}%)")
        positions = adapter.fetch_positions()
        target_positions = [p for p in positions if p["pair_index"] == pair_index]
        if not target_positions:
            print(f"  没有找到 {args.symbol} 的持仓")
            return 0

        print(f"  {args.symbol} 共 {len(target_positions)} 个持仓:")
        for p in target_positions:
            side = "LONG" if p["long"] else "SHORT"
            col = p["collateral_amount_raw"] / 10**6
            lev = p["leverage"]
            print(f"    index={p['index']}  {side}  {col:.2f} USDC  {lev:.1f}x")

        if not args.execute:
            print(f"\ndry-run 模式：加 --execute 后才会逐个发送平仓交易")
            return 0

        total = len(target_positions)
        for i, p in enumerate(target_positions, 1):
            print(f"\n[{i}/{total}] 平仓 {args.symbol} index={p['index']}...", end=" ")
            req = OnchainCloseRequest(
                symbol=args.symbol,
                position_id=str(p["index"]),
                slippage=args.close_slippage,
            )
            try:
                result = adapter.close_trade(req)
                print(f"✅ tx={result.tx_hash}")
            except Exception as e:
                print(f"❌ 失败: {e}")
        return 0

    print(f"操作:   开 {args.side.upper()} {args.symbol}")
    print(f"  抵押品: {args.collateral} USDC")
    print(f"  杠杆:   {args.leverage}x")
    print(f"  仓位值: ${args.collateral * args.leverage:.0f} USD")
    print(f"  slippage: {args.slippage*100:.1f}%")
    print(f"  pairIndex: {pair_index}")
    print(f"  RPC:   {config.rpc_url}")
    print(f"  钱包:  {WALLET_ADDRESS[:10]}...{WALLET_ADDRESS[-6:]}")
    _print_gas_info(adapter, is_close=False)
    _print_usdc_allowance(adapter, args.collateral)

    if not args.execute:
        print(f"\ndry-run 模式：加 --execute 后才会发送链上交易")
        return 0

    eth_balance = _print_gas_info(adapter, is_close=False)
    w3 = adapter._get_web3()
    gas_params = adapter._build_gas_params(w3)
    max_fee = gas_params.get("maxFeePerGas") or w3.eth.gas_price
    if eth_balance < GAS_LIMIT_OPEN_TRADE * max_fee / 10**18:
        print("\n错误: ETH 余额不足以支付 gas，取消交易")
        return 1

    req = OnchainOpenRequest(
        symbol=args.symbol,
        side=args.side,
        collateral=args.collateral,
        leverage=args.leverage,
        slippage=args.slippage,
    )
    result = adapter.open_trade(req)
    print(f"开仓成功: tx={result.tx_hash}  trade_index={result.order_sys_id}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        
        # # 开仓
        # sys.argv = [
        #     __file__,
        #     "--symbol", "BTC",
        #     "--side", "long",
        #     "--collateral", "2",
        #     "--leverage", "1.1",
        #     "--slippage", "0.01",
        #     "--execute",
        # ]

        # 平掉一个symbol的所有仓位
        sys.argv = [
            __file__,
            "--close-symbol",   # 平掉 BTC 全部仓位
            "--symbol", "BTC",
            "--close-slippage", "0.01",
            "--execute",
        ]

        # ===== 备用默认参数（取消注释即可切换） =====
        # # 平掉所有symbol的所有仓位
        # sys.argv = [
        #     __file__,
        #     "--close-all",
        #     "--close-slippage", "0.01",
        #     "--execute",
        # ]
    sys.exit(main())
