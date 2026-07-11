MY-BACKTEST4 — CHẠY NGAY

CÁCH NHANH NHẤT
1. Copy đúng 1 file BTC 15m .parquet vào folder repo.

2. Double-click:
   RUN_NOW.cmd

Hoặc PowerShell:
   cd C:\phong\AI\finance\my-backtest4
   .\run.ps1 -Data "C:\duong-dan\btc_15m.parquet"

KẾT QUẢ
   output\FINAL_RESULT.txt
   output\01_train_all.csv
   output\02_validation_shortlist.csv
   output\03_final_oos.csv

Không phải lần chạy nào cũng tạo đủ 3 file kết quả:
- Nếu TRAIN không đạt gate, runner dừng tại TRAIN.
- Nếu VALIDATION không đạt gate, runner dừng trước FINAL OOS.
- FINAL OOS chỉ được mở khi candidate đã qua cả TRAIN và VALIDATION.

GATE MẶC ĐỊNH
- TRAIN: expectancy >= +0.225R và ít nhất 300 lệnh.
- VALIDATION: expectancy >= +0.225R và ít nhất 50 lệnh.
- FINAL OOS: expectancy >= +0.225R và ít nhất 50 lệnh.

DATA TỐI THIỂU
- DatetimeIndex, hoặc cột:
  datetime / date / timestamp / time / open_time
- Cột:
  open, high, low, close
- volume không bắt buộc
- Script tự chuẩn hóa datetime sang UTC.

MẶC ĐỊNH CHI PHÍ
- Fee mỗi fill:       0.0005 = 0.05%
- Slippage mỗi fill:  0.0002 = 0.02%

Muốn đổi gate:
   .\.venv\Scripts\python.exe .\search_gate.py `
     --data "C:\duong-dan\btc.parquet" `
     --target-r 0.225 `
     --min-train-trades 300 `
     --min-validation-trades 50 `
     --min-oos-trades 50

MODE
- instant: quét nhanh 3 family:
  Donchian breakout, RSI pullback, Bollinger re-entry
- full: grid lớn hơn nhiều.

LƯU Ý
- Signal tính ở close và entry ở open nến kế tiếp.
- Expectancy được tính bằng Net PnL / Initial Risk.
- Fee và slippage đã nằm trong Net PnL.
- Bollinger config trùng do EMA không được dùng sẽ bị loại tự động.
- Candidate cuối vẫn phải chạy Exact Engine để khóa same-bar/both-hit behavior.
