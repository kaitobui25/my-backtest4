import sys

import pytest

from btsearch.report import safe_print


class _Cp932Stdout:
    """Mimics a Windows cp932 console: cannot encode characters such as the
    em-dash or Vietnamese letters, raising UnicodeEncodeError like the real
    C runtime does."""

    encoding = "cp932"

    def write(self, s: str) -> int:
        s.encode("cp932")
        return len(s)

    def flush(self) -> None:
        pass


def test_safe_print_handles_non_utf8_console(monkeypatch):
    fake = _Cp932Stdout()
    monkeypatch.setattr(sys, "stdout", fake)

    report = "Báo cáo — MY-BACKTEST4 — Kết quả: +0.225R"
    with pytest.raises(UnicodeEncodeError):
        print(report)

    safe_print(report)
