"""Demo Structured Streaming w PySpark.

Skrypt obserwuje katalog `stream_input` i agreguje sprzedaż per kraj.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.types import StructType, StringType, DoubleType


def main() -> None:
    """Uruchamia demo streamingu."""
    csv_path = Path("/content/")
    input_dir = csv_path / "stream_input"
    checkpoint_dir = csv_path / "stream_checkpoint"

    input_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("BigDataBlock1SparkStreaming")
        .master("local[*]")
        .getOrCreate()
    )

    schema = (
        StructType()
        .add("order_id", StringType())
        .add("country", StringType())
        .add("category", StringType())
        .add("amount", DoubleType())
        .add("order_date", StringType())
    )

    stream_df = (
        spark.readStream.schema(schema)
        .option("sep", ",")
        .csv(str(input_dir))
    )

    sales_by_country = stream_df.groupBy("country").agg(
        spark_sum("amount").alias("total_sales")
    )

    query = (
        sales_by_country.writeStream.outputMode("complete")
        .format("console")
        .option("truncate", False)
        .option("checkpointLocation", str(checkpoint_dir))
        .start()
    )

    print("Streaming started. Now run: python scripts/generate_streaming_files.py")
    query.awaitTermination()


if __name__ == "__main__":
    main()
