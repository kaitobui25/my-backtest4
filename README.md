# MY-BACKTEST4 — VECTORBT STRATEGY RESEARCH ENGINE

Engine nghiên cứu BTC Futures 15m, dùng vectorbt để quét hàng nghìn config theo batch.

## Chạy ngay

1. Giải nén vào:

```text
C:\phong\AI\finance\my-backtest4
```

2. Copy file BTC 15m `.parquet` vào folder.

3. Chạy:

```powershell
cd C:\phong\AI\finance\my-backtest4

.\run.ps1 `
  -Data "btcusdt_futures_15m_20210101_to_20260711.parquet" `
  -Mode coarse
```

Tiếp tục phiên bị dừng:

```powershell
.\run.ps1 -Data "..." -Mode coarse -Resume
```

Full search:

```powershell
.\run.ps1 -Data "..." -Mode full -Resume
```

Test nhanh 500 config:

```powershell
.\run.ps1 -Data "..." -Mode coarse -MaxConfigs 500
```

## Strategy families

- Donchian breakout
- Rolling range breakout
- EMA crossover
- EMA slope pullback
- RSI momentum
- ROC momentum
- MACD momentum
- Bollinger re-entry
- Bollinger fade
- Z-score mean reversion
- RSI extreme mean reversion
- ATR expansion breakout
- Low-volatility squeeze breakout
- Breakout + volume
- Momentum + volume
- Session filters refined around top candidates: Asia, London, New York, overlap, high-liquidity UTC
- Weekday filters refined around top candidates

Volume strategies tự skip nếu data không có volume.

## Nguyên tắc

- Signal xác nhận tại close.
- Entry tại open candle kế tiếp.
- Rolling levels không nhìn candle hiện tại.
- Fee và slippage được tính trong net PnL.
- `R = net trade PnL / initial risk cash`.
- Config ít lệnh vẫn được lưu nhưng không đứng đầu ranking chính.
- Search không dừng chỉ vì chưa đạt `+0.225R`.
- Final OOS cũ không được coi là pristine; engine dùng expanding walk-forward.

## Output

```text
output/
├── 01_phase1_all_results.parquet
├── 01_phase1_all_results.csv
├── 01_phase1_insufficient_sample.csv
├── 02_phase1_top_by_family.csv
├── 03_phase2_refined_results.parquet
├── 03_phase2_refined_results.csv
├── 04_walk_forward_results.parquet
├── 04_walk_forward_results.csv
├── 05_top_expectancy.csv
├── 06_top_robust.csv
├── BEST_EXPECTANCY_CONFIG.json
├── BEST_ROBUST_CONFIG.json
└── FINAL_REPORT.txt
```

## Data tối thiểu

DatetimeIndex hoặc một cột:

```text
datetime / date / timestamp / time / open_time
```

OHLC:

```text
open, high, low, close
```

`volume` không bắt buộc.

## Chi phí mặc định

```text
fee mỗi fill      = 0.0005
slippage mỗi fill = 0.0002
```

Override:

```powershell
.\run.ps1 -Data "..." -Fee 0.0004 -Slippage 0.0001
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Lưu ý quan trọng

Đây là fast research engine. Candidate cuối vẫn phải chạy lại bằng Exact Engine để khóa:

- same-bar behavior
- cả SL và TP cùng bị chạm
- funding
- exchange-specific order behavior
