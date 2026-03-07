import os
"""Analiza danych sprzedażowych w PySpark batch."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, sum as spark_sum, desc


def main() -> None:
    """Uruchamia analizę wsadową w Spark."""
    # root_dir = Path(__file__).resolve().parents[1] # This line caused the NameError
    csv_path = Path("/content/data/sales.csv") # Directly specify the path

    spark = (
        SparkSession.builder.appName("BigDataBlock1SparkBatch")
        .master("local[*]")
        .getOrCreate()
    )

    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)

    print("\n=== SPARK BATCH DEMO ===")
    print("\n1. Schemat danych:")
    df.printSchema()

    print("\n2. Podgląd danych:")
    df.show(5, truncate=False)

    print("\n3. Liczba rekordów:")
    print(df.count())

    print("\n4. Średnia wartość zamówienia:")
    df.select(avg("amount").alias("avg_amount")).show()

    print("\n5. Suma sprzedaży per kraj:")
    sales_by_country = df.groupBy("country").agg(spark_sum("amount").alias("total_sales"))
    sales_by_country.orderBy(desc("total_sales")).show(truncate=False)

    print("\n6. Kraj z największą sprzedażą:")
    sales_by_country.orderBy(desc("total_sales")).show(1, truncate=False)

    spark.stop()

main()
