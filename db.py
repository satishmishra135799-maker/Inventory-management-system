"""
db.py — SQLite database for products and sales history.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'store.db')

SEED_PRODUCTS = [
    ('Ariel',           299, 50),
    ('Coca Cola',        40, 100),
    ('Colgate',          89, 80),
    ('Fanta',            40, 100),
    ('Kurkure',          20, 150),
    ('Lays Masala',      20, 150),
    ('Lays Mexican',     20, 150),
    ('Lifebuoy Soap',    55, 80),
    ('Sunsilk Shampoo', 179, 60),
    ('Vaseline Lotion', 199, 60),
]


# Variants: (brand, label, price)
SEED_VARIANTS = [
    ('Ariel',           'Ariel 500g',          199),
    ('Ariel',           'Ariel 1kg',            399),
    ('Ariel',           'Ariel 2kg',            699),
    ('Coca Cola',       'Coca Cola 250ml',       20),
    ('Coca Cola',       'Coca Cola 500ml',       40),
    ('Coca Cola',       'Coca Cola 1.25L',       80),
    ('Colgate',         'Colgate 50g',           45),
    ('Colgate',         'Colgate 100g',          89),
    ('Colgate',         'Colgate 200g',         149),
    ('Fanta',           'Fanta 250ml',           20),
    ('Fanta',           'Fanta 500ml',           40),
    ('Kurkure',         'Kurkure 22g',           10),
    ('Kurkure',         'Kurkure 45g',           20),
    ('Kurkure',         'Kurkure 90g',           40),
    ('Lays Masala',     'Lays Masala 26g',       10),
    ('Lays Masala',     'Lays Masala 52g',       20),
    ('Lays Mexican',    'Lays Mexican 26g',      10),
    ('Lays Mexican',    'Lays Mexican 52g',      20),
    ('Lifebuoy Soap',   'Lifebuoy 100g',         35),
    ('Lifebuoy Soap',   'Lifebuoy 150g',         55),
    ('Sunsilk Shampoo', 'Sunsilk 180ml',         99),
    ('Sunsilk Shampoo', 'Sunsilk 340ml',        179),
    ('Sunsilk Shampoo', 'Sunsilk 650ml',        299),
    ('Vaseline Lotion', 'Vaseline 100ml',        99),
    ('Vaseline Lotion', 'Vaseline 200ml',       199),
    ('Vaseline Lotion', 'Vaseline 400ml',       349),
]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT    UNIQUE NOT NULL,
                price   REAL    NOT NULL,
                stock   INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS variants (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                brand   TEXT NOT NULL,
                label   TEXT UNIQUE NOT NULL,
                price   REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sales (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                product    TEXT    NOT NULL,
                qty        INTEGER NOT NULL,
                unit_price REAL    NOT NULL,
                subtotal   REAL    NOT NULL
            );
        """)
        # Seed products if table is empty
        if conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO products (name, price, stock) VALUES (?,?,?)",
                SEED_PRODUCTS
            )
        if conn.execute("SELECT COUNT(*) FROM variants").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO variants (brand, label, price) VALUES (?,?,?)",
                SEED_VARIANTS
            )


def get_variants(brand: str) -> list[dict]:
    """Return all size variants for a detected brand."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT label, price FROM variants WHERE brand=? ORDER BY price",
            (brand,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_products() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM products ORDER BY name").fetchall()]


def get_price_db() -> dict:
    """Return {name: price} dict — used by billing.py."""
    return {r['name']: r['price'] for r in get_products()}


def upsert_product(name: str, price: float, stock: int):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO products (name, price, stock) VALUES (?,?,?)
            ON CONFLICT(name) DO UPDATE SET price=excluded.price, stock=excluded.stock
        """, (name, price, stock))


def delete_product(name: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM products WHERE name=?", (name,))


def record_sale(cart: dict, price_db: dict):
    """Save each line item of a completed sale."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for product, qty in cart.items():
        unit = price_db.get(product, 0)
        rows.append((ts, product, qty, unit, unit * qty))
        # Deduct stock
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO sales (ts,product,qty,unit_price,subtotal) VALUES (?,?,?,?,?)",
            rows
        )
        for product, qty in cart.items():
            conn.execute(
                "UPDATE products SET stock = MAX(0, stock-?) WHERE name=?",
                (qty, product)
            )


def sales_summary(days: int = 7) -> list[dict]:
    """Revenue per product over last N days."""
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("""
            SELECT product,
                   SUM(qty)      AS total_qty,
                   SUM(subtotal) AS revenue
            FROM   sales
            WHERE  ts >= datetime('now', ?)
            GROUP  BY product
            ORDER  BY revenue DESC
        """, (f'-{days} days',)).fetchall()]


def daily_revenue(days: int = 7) -> list[dict]:
    """Total revenue per day over last N days."""
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("""
            SELECT DATE(ts) AS day, SUM(subtotal) AS revenue
            FROM   sales
            WHERE  ts >= datetime('now', ?)
            GROUP  BY day
            ORDER  BY day
        """, (f'-{days} days',)).fetchall()]


# Initialise on import
init_db()
