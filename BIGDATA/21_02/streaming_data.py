from __future__ import annotations

import os
import tempfile
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    spark = (
        SparkSession.builder
        .appName("PyCharm-Structured-Streaming-Demo")
        .master("local[*]")
        # dla czytelności logów na zajęciach:
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Checkpoint w temp (ważne w streamingu, Spark tego lubi)
    checkpoint_dir = os.path.join(tempfile.gettempdir(), "spark_streaming_demo_checkpoint")

    # 1) Źródło: rate (Spark sam generuje eventy)
    # - rowsPerSecond: ile rekordów na sekundę
    # - każda linia ma: timestamp, value (rosnący long)
    raw = (
        spark.readStream
        .format("rate")
        .option("rowsPerSecond", 20)
        .option("rampUpTime", 0)
        .load()
    )

    # 2) Transformacja: z "value" robimy sztuczne eventy zakupowe
    # - user_id: 0..999
    # - amount: 1.00..100.99 (deterministycznie z value)
    # - country: PL/DE/CZ/SK (deterministycznie z value)
    events = (
        raw
        .withColumn("user_id", (F.col("value") % F.lit(1000)).cast("int"))
        .withColumn("amount", (F.col("value") % F.lit(10000) / F.lit(100.0) + F.lit(1.0)).cast("double"))
        .withColumn(
            "country",
            F.element_at(
                F.array(F.lit("PL"), F.lit("DE"), F.lit("CZ"), F.lit("SK")),
                (F.col("value") % F.lit(4) + F.lit(1)).cast("int")  # element_at jest 1-indexed
            )
        )
        .select("timestamp", "user_id", "amount", "country")
    )

    # 3) Agregacja w oknach czasowych (windowed aggregation)
    # - okno 10 sekund, przesuwane co 5 sekund
    # - sum(amount), avg(amount), count
    agg = (
        events
        .withWatermark("timestamp", "20 seconds")  # „pamięć” na spóźnione eventy
        .groupBy(
            F.window("timestamp", "10 seconds", "5 seconds"),
            F.col("country")
        )
        .agg(
            F.count("*").alias("events"),
            F.round(F.sum("amount"), 2).alias("revenue"),
            F.round(F.avg("amount"), 2).alias("avg_amount"),
        )
        .orderBy(F.col("window").asc(), F.col("country").asc())
    )

    # 4) Wyjście: konsola
    # outputMode:
    # - "update": pokazuje zmieniające się wyniki w oknach
    # - "complete": drukuje całą tabelę za każdym batch (bardziej „efektownie”, ale cięższe)
    query = (
        agg.writeStream
        .format("console")
        .outputMode("update")
        .option("truncate", "false")
        .option("numRows", 200)
        .option("checkpointLocation", checkpoint_dir)
        .start()
    )

    print("\nStreaming wystartował. Poczekaj ~25s, potem sam się zakończy.\n")
    query.awaitTermination(25)

    query.stop()
    spark.stop()
    print("\nZakończone.\n")


if __name__ == "__main__":
    main()
