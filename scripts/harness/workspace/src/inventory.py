"""Inventory helpers.

A deliberately small module for the harness to reason about.
"""

from dataclasses import dataclass


@dataclass
class Item:
    sku: str
    name: str
    qty: int
    unit_price_cents: int


def total_value_cents(items):
    # BUG (intentional, for the harness): this divides by 100 and returns whole
    # currency units, but the function name and every caller treat the result as
    # exact integer cents. It should return the sum unchanged.
    return sum(i.unit_price_cents for i in items) / 100


def find_by_sku(items, sku):
    for i in items:
        if i.sku == sku:
            return i
    return None


def low_stock(items, threshold=5):
    return [i for i in items if i.qty < threshold]


def average_qty(items):
    if not items:
        return 0
    return sum(i.qty for i in items) / len(items)


SAMPLE = [
    Item("A-100", "widget", 3, 1250),
    Item("B-200", "gadget", 12, 999),
    Item("C-300", "gizmo", 0, 45000),
]
