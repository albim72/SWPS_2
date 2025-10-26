#!/usr/bin/env python3
"""
generate_svm_data.py
--------------------
Generates synthetic CSV data for nonlinear SVM experiments.

Columns:
  - id (int)                      : sequential row id starting at 1
  - wartosc (float)               : random value in (-10, 102)
  - miasto (str)                  : Polish city (Faker, locale pl_PL)
  - mieszkancy (int)              : random integer in [50, 15600]
  - kolor (str)                   : categorical with probs: czerwony 70%, zielony 25%, żółty 5%

Usage:
  python generate_svm_data.py --n 5000000 --out dane_svm_5mln.csv --chunksize 250000 --seed 42

Notes:
  * Uses chunked generation to keep memory usage moderate.
  * Install dependencies:
      pip install numpy pandas faker
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional
import numpy as np
import pandas as pd
from faker import Faker

def gen_chunk(start_id: int, size: int, faker: Faker, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a single chunk of rows."""
    ids = np.arange(start_id, start_id + size, dtype=np.int64)

    # wartosc: uniform in (-10, 102)
    wartosc = rng.uniform(-10.0, 102.0, size)

    # miasto: Faker city (vectorized via list-comp due to Faker API)
    miasto = [faker.city() for _ in range(size)]

    # mieszkancy: inclusive bounds [50, 15600]
    mieszkancy = rng.integers(50, 15601, size=size, dtype=np.int32)

    # kolor with given probabilities
    kolory = np.array(["czerwony", "zielony", "żółty"], dtype=object)
    kolor = rng.choice(kolory, size=size, p=[0.70, 0.25, 0.05])

    df = pd.DataFrame({
        "id": ids,
        "wartosc": wartosc,
        "miasto": miasto,
        "mieszkancy": mieszkancy,
        "kolor": kolor,
    })
    return df

def main(n: int, out: str, chunksize: int, seed: Optional[int], locale: str, gzip: bool) -> None:
    if n <= 0:
        raise SystemExit("Parametr --n musi być dodatni.")
    if chunksize <= 0:
        raise SystemExit("Parametr --chunksize musi być dodatni.")
    if chunksize > n:
        chunksize = n

    faker = Faker(locale)
    rng = np.random.default_rng(seed)

    # Prepare writer (write header on first chunk, then append without header)
    written = 0
    next_id = 1
    mode = "wt"
    compression = "gzip" if gzip else None

    while written < n:
        size = min(chunksize, n - written)
        df = gen_chunk(next_id, size, faker, rng)

        df.to_csv(out, index=False, mode=mode, header=(written == 0), compression=compression)
        written += size
        next_id += size
        mode = "at"  # append for subsequent chunks

    print(f"Zapisano {written:,} rekordów do pliku: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator danych syntetycznych dla SVM (PL).")
    parser.add_argument("--n", type=int, default=5_000_000, help="Liczba rekordów do wygenerowania (domyślnie 5,000,000).")
    parser.add_argument("--out", type=str, default="dane_svm_5mln.csv", help="Ścieżka wyjściowa do pliku CSV.")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Wielkość porcji generacji (domyślnie 250k).")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno RNG dla powtarzalności (domyślnie 42).")
    parser.add_argument("--locale", type=str, default="pl_PL", help="Locale dla Faker (domyślnie pl_PL).")
    parser.add_argument("--gzip", action="store_true", help="Jeśli podane, zapisze plik skompresowany .gz")

    args = parser.parse_args()
    try:
        main(n=args.n, out=args.out, chunksize=args.chunksize, seed=args.seed, locale=args.locale, gzip=args.gzip)
    except KeyboardInterrupt:
        sys.exit("Przerwano przez użytkownika.")
