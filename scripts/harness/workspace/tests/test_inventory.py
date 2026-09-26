"""Tests for inventory helpers."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inventory import Item, total_value_cents, find_by_sku, low_stock, average_qty, SAMPLE


def test_total_value():
    # 1250 + 999 + 45000 = 47249 cents
    assert total_value_cents(SAMPLE) == 47249, total_value_cents(SAMPLE)


def test_find_by_sku():
    assert find_by_sku(SAMPLE, "B-200").name == "gadget"
    assert find_by_sku(SAMPLE, "nope") is None


def test_low_stock():
    skus = sorted(i.sku for i in low_stock(SAMPLE, threshold=5))
    assert skus == ["A-100", "C-300"], skus


def test_average_qty():
    assert average_qty(SAMPLE) == 5.0, average_qty(SAMPLE)
    assert average_qty([]) == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
