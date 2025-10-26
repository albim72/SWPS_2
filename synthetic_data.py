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

# Call main directly with desired arguments
try:
    main(n=5000000, out="dane_svm_5mln.csv", chunksize=250000, seed=42, locale="pl_PL", gzip=False)
except KeyboardInterrupt:
    sys.exit("Przerwano przez użytkownika.")
