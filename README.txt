MY-BACKTEST4 — CHẠY NGAY

CÁCH NHANH NHẤT
1. Giải nén toàn bộ vào:
   C:\phong\AI\finance\my-backtest4

2. Copy đúng 1 file BTC 15m .parquet vào folder trên.

3. Double-click:
   RUN_NOW.cmd

Hoặc PowerShell:
   cd C:\phong\AI\finance\my-backtest4
   .\run.ps1 -Data "C:\duong-dan\btc_15m.parquet"

KẾT QUẢ
   output\FINAL_RESULT.txt
   output\BEST_FROZEN_CONFIG.json
   output\01_train_all.csv
   output\02_validation_shortlist.csv
   output\03_final_oos.csv

DATA TỐI THIỂU
- DatetimeIndex, hoặc cột:
  datetime / date / timestamp / time / open_time
- Cột:
  open, high, low, close
- volume không bắt buộc
- Script tự chuẩn hóa datetime sang UTC.

MẶC ĐỊNH CHI PHÍ
- Fee mỗi fill:     0.0005 = 0.05%
- Slippage mỗi fill: 0.0002 = 0.02%

Muốn đổi:
   .\.venv\Scripts\python.exe .\search.py `
     --data "C:\duong-dan\btc.parquet" `
     --fee 0.0004 `
     --slippage 0.0001 `
     --target-r 0.225

MODE
- instant: quét nhanh 3 family:
  Donchian breakout, RSI pullback, Bollinger re-entry
- full: grid rất lớn, không còn là "3 nốt nhạc"

LƯU Ý
- Signal tính ở close và entry ở open nến kế tiếp.
- Expectancy được tính bằng Net PnL / Initial Risk.
- Fee và slippage đã nằm trong Net PnL.
- Đây là fast screening. Candidate cuối vẫn phải chạy Exact Engine
  để khóa same-bar/both-hit behavior trước khi trade thật.
