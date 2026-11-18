"""
Load and clean Online Retail II (UCI) into a daily demand table.

Source, in order of preference:
  1. data/raw/uci_original/online_retail_II.xlsx  (official UCI file, both sheets)
  2. GitHub mirrors in data/raw/ (online_retail_II.csv for 2009-10, onlineretail.rda
     for 2010-11). The mirrors match the official file except one row (2010-11
     sheet has 541,910 rows vs 541,909 in the mirror).

The two sheets overlap on 1-9 Dec 2010 (sheet 1 runs to 9 Dec 2010, sheet 2
starts 1 Dec 2010), so sheet 1 is cut at 30 Nov 2010 to avoid double counting.

Cleaning:
  * keep real product lines (codes starting with 5 digits); drop postage, fees,
    manual adjustments, bank charges
  * drop zero/negative prices
  * cancellations (invoice starting 'C', or negative quantity) remove the original
    order line they reverse (same customer, product, quantity, earlier or same time)
Outputs:
  data/daily_demand.parquet  date, StockCode, qty, revenue
  data/products.parquet      StockCode, description, median_price, price, units
     price = volume-weighted average unit price actually paid (revenue / units)
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OFFICIAL = RAW / "uci_original" / "online_retail_II.xlsx"
OFFICIAL_CACHE = RAW / "uci_original" / "online_retail_II_raw.parquet"
COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
        "Price", "CustomerID", "Country"]


def _load_official():
    if OFFICIAL_CACHE.exists():
        return pd.read_parquet(OFFICIAL_CACHE), "official UCI xlsx (cached)"
    sheets = pd.read_excel(OFFICIAL, sheet_name=None)
    a, b = list(sheets.values())
    for d in (a, b):
        d.columns = COLS
    df = pd.concat([a[a.InvoiceDate < "2010-12-01"], b], ignore_index=True)
    for c in ("Invoice", "StockCode", "Description", "Country"):
        df[c] = df[c].astype(str)
    df.to_parquet(OFFICIAL_CACHE)
    return df, "official UCI xlsx"


def _load_mirrors():
    import pyreadr
    a = pd.read_csv(RAW / "online_retail_II.csv", encoding="utf-8-sig")
    a.columns = COLS
    a["InvoiceDate"] = pd.to_datetime(a["InvoiceDate"], format="%d/%m/%Y %H:%M")
    b = pyreadr.read_r(RAW / "onlineretail.rda")["onlineretail"]
    b.columns = COLS
    b["InvoiceDate"] = pd.to_datetime(b["InvoiceDate"])
    return pd.concat([a[a.InvoiceDate < "2010-12-01"], b], ignore_index=True), "GitHub mirrors"


def load_raw():
    df, src = _load_official() if (OFFICIAL.exists() or OFFICIAL_CACHE.exists()) else _load_mirrors()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.upper()
    df.attrs["source"] = src
    return df


def net_cancellations(sales, cancels):
    """Remove, for each cancellation, the latest earlier order line with the same
    customer, product and quantity. Returns (kept sales, number removed)."""
    sales = sales.copy()
    cancels = cancels.copy()
    sales["key"] = (sales.CustomerID.astype(str) + "|" + sales.StockCode + "|"
                    + sales.Quantity.astype(str))
    cancels["key"] = (cancels.CustomerID.astype(str) + "|" + cancels.StockCode + "|"
                      + (-cancels.Quantity).astype(str))
    sales = sales.sort_values("InvoiceDate", kind="stable").reset_index(drop=True)
    drop = set()
    by_key = sales.groupby("key").indices
    dates = sales.InvoiceDate.to_numpy()
    for key, when in zip(cancels.key, cancels.InvoiceDate.to_numpy()):
        idx = by_key.get(key)
        if idx is None:
            continue
        cand = [i for i in idx if i not in drop and dates[i] <= when]
        if cand:
            drop.add(cand[-1])
    return sales.drop(index=list(drop)).drop(columns="key"), len(drop)


def clean(df, verbose=True):
    df = df[df.StockCode.str.match(r"^\d{5}")].copy()
    df = df[df.Price > 0]
    is_cancel = df.Invoice.str.startswith("C") | (df.Quantity < 0)
    sales, removed = net_cancellations(df[~is_cancel], df[is_cancel])
    if verbose:
        print(f"removed {removed:,} order lines that were later cancelled")
    return sales


def main():
    raw = load_raw()
    s = clean(raw)
    s["date"] = s.InvoiceDate.dt.normalize()
    s["revenue"] = s.Quantity * s.Price
    daily = (s.groupby(["date", "StockCode"])
               .agg(qty=("Quantity", "sum"), revenue=("revenue", "sum"))
               .reset_index())
    prods = (s.groupby("StockCode")
               .agg(description=("Description", lambda x: x.mode().iat[0]),
                    median_price=("Price", "median"), units=("Quantity", "sum"),
                    revenue=("revenue", "sum"))
               .reset_index())
    prods["price"] = prods.revenue / prods.units          # volume-weighted unit price
    daily.to_parquet(ROOT / "data" / "daily_demand.parquet")
    prods.to_parquet(ROOT / "data" / "products.parquet")
    print(f"source: {raw.attrs['source']}; raw rows {len(raw):,}")
    print(f"{len(s):,} clean order lines, {daily.StockCode.nunique():,} products, "
          f"{daily.date.min().date()} to {daily.date.max().date()}")
    r = (prods.price / prods.median_price).describe()
    print(f"weighted price / median list price: median {r['50%']:.3f}, IQR {r['25%']:.3f}-{r['75%']:.3f}")


if __name__ == "__main__":
    main()
