"""Generator plików CSV do demonstracji Structured Streaming.

Skrypt co kilka sekund dopisuje nowy plik CSV do katalogu `stream_input`.
"""

from __future__ import annotations

import csv
import random
import time
from datetime import datetime
from pathlib import Path


COUNTRIES = ["Poland", "Germany", "France", "Spain", "Italy", "Czechia"]
CATEGORIES = ["Books", "Electronics", "Clothing", "Sports", "Beauty"]


def write_batch(batch_no: int, output_dir: Path, rows: int = 15) -> None:
    """Zapisuje pojedynczy plik z porcją zdarzeń sprzedażowych."""
    file_path = output_dir / f"batch_{batch_no:03d}.csv"

    with file_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for idx in range(rows):
            order_id = f"STR-{batch_no:03d}-{idx:03d}"
            country = random.choice(COUNTRIES)
            category = random.choice(CATEGORIES)
            amount = round(random.uniform(10.0, 900.0), 2)
            order_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([order_id, country, category, amount, order_date])

    print(f"Written: {file_path.name}")


def main() -> None:
    """Generuje kolejne pliki wejściowe dla streamingu."""
    random.seed(123)

    # root_dir = Path(__file__).resolve().parents[1]
    csv_path = Path("/content/")
    output_dir = csv_path / "stream_input"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating streaming files...")
    for batch_no in range(1, 6):
        write_batch(batch_no=batch_no, output_dir=output_dir, rows=15)
        time.sleep(3)

    print("Done.")


main()
