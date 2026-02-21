"""
dynamic_spark_processor.py

Metaprogramowanie + PySpark:
- generowanie "kontraktu danych" (StructType) z definicji schematu
- schema enforcement przez from_json / cast (bez UDF, szybciej)
- opcjonalne UDF: normalizacja + walidacja rekordu (gdy logika niestandardowa)

Uruchom:
  spark-submit dynamic_spark_processor.py
albo lokalnie:
  python dynamic_spark_processor.py   (jeśli masz pyspark)

Wymagania:
  pip install pyspark
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ---------------------------
# 1) Definicja kontraktu (FieldSpec) + mapowanie typów
# ---------------------------

@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: T.DataType
    required: bool = True

    # "coercion" i "validator" są przeznaczone do wersji UDF (opcjonalne)
    coercion: Optional[Callable[[Any], Any]] = None
    validator: Optional[Callable[[Any], Optional[str]]] = None  # None jeśli OK, albo string z błędem


def string(name: str, required: bool = True, *, coercion=None, validator=None) -> FieldSpec:
    return FieldSpec(name=name, dtype=T.StringType(), required=required, coercion=coercion, validator=validator)

def integer(name: str, required: bool = True, *, coercion=None, validator=None) -> FieldSpec:
    return FieldSpec(name=name, dtype=T.IntegerType(), required=required, coercion=coercion, validator=validator)

def double(name: str, required: bool = True, *, coercion=None, validator=None) -> FieldSpec:
    return FieldSpec(name=name, dtype=T.DoubleType(), required=required, coercion=coercion, validator=validator)

def timestamp(name: str, required: bool = True, *, coercion=None, validator=None) -> FieldSpec:
    return FieldSpec(name=name, dtype=T.TimestampType(), required=required, coercion=coercion, validator=validator)


def build_struct_schema(fields: Iterable[FieldSpec]) -> T.StructType:
    """Schema enforcement: StructType dla from_json / selectExpr / cast."""
    return T.StructType([T.StructField(f.name, f.dtype, nullable=not f.required) for f in fields])


# ---------------------------
# 2) Metaprogramowanie: fabryka procesora Spark
# ---------------------------

def create_spark_processor(
    name: str,
    fields: List[FieldSpec],
    *,
    input_col: str = "raw_json",
    output_col: str = "data",
    keep_raw: bool = True,
):
    """
    Zwraca klasę procesora, która:
    - posiada schema (StructType)
    - method enforce_schema(df): schema enforcement + flagowanie braków
    - method with_validation_udf(df): (opcjonalnie) UDF walidujący/normalizujący rekord
    """
    struct_schema = build_struct_schema(fields)
    required_fields = [f.name for f in fields if f.required]

    # --- (A) fast path: schema enforcement bez UDF ---
    def enforce_schema(self, df: DataFrame) -> DataFrame:
        """
        Zakładamy, że wejście ma kolumnę JSON string (input_col).
        Parsujemy ją do struct o zadanym schemacie (nullable wg required),
        a potem oznaczamy rekordy niezgodne (np. brak wymaganych pól).
        """
        parsed = df.withColumn(output_col, F.from_json(F.col(input_col), self.schema))

        missing_exprs = [F.col(f"{output_col}.{c}").isNull() for c in required_fields]
        if missing_exprs:
            is_missing = missing_exprs[0]
            for e in missing_exprs[1:]:
                is_missing = is_missing | e
        else:
            is_missing = F.lit(False)

        parsed = parsed.withColumn(
            "is_schema_ok",
            (~is_missing) & F.col(output_col).isNotNull()
        ).withColumn(
            "schema_error",
            F.when(F.col(output_col).isNull(), F.lit("JSON_PARSE_FAILED"))
             .when(is_missing, F.lit("MISSING_REQUIRED_FIELD_OR_TYPE_CAST"))
             .otherwise(F.lit(None))
        )

        if not keep_raw:
            parsed = parsed.drop(input_col)

        return parsed

    # --- (B) opcjonalna walidacja/normalizacja przez UDF ---
    def _validate_and_normalize_py(record: Any) -> Tuple[Optional[dict], bool, Optional[str]]:
        """
        record: dict-like (z parsowanego struct)
        returns: (normalized_dict_or_none, is_valid, error_or_none)

        normalized_dict jest mapą field->string (dla prostoty), łatwą do logowania/debugu.
        """
        if record is None:
            return None, False, "NO_DATA"

        # Spark Row -> dict
        if isinstance(record, Row):
            rec = record.asDict(recursive=True)
        elif isinstance(record, dict):
            rec = record
        else:
            try:
                rec = dict(record)
            except Exception:
                return None, False, "UNSUPPORTED_RECORD_TYPE"

        # required check
        for f in fields:
            if f.required and (f.name not in rec or rec[f.name] is None):
                return None, False, f"MISSING:{f.name}"

        out: Dict[str, Any] = {}
        for f in fields:
            v = rec.get(f.name, None)

            if f.coercion is not None:
                try:
                    v = f.coercion(v)
                except Exception:
                    return None, False, f"COERCION_FAILED:{f.name}"

            # dodatkowa ostrożna koercja po pythonowej stronie (best-effort)
            if v is not None:
                try:
                    if isinstance(f.dtype, T.IntegerType):
                        v = int(v)
                    elif isinstance(f.dtype, T.DoubleType):
                        v = float(v)
                    elif isinstance(f.dtype, T.StringType):
                        v = str(v)
                except Exception:
                    return None, False, f"TYPE_CAST_FAILED:{f.name}"

            if f.validator is not None:
                err = f.validator(v)
                if err:
                    return None, False, f"VALIDATION_FAILED:{f.name}:{err}"

            out[f.name] = v

        # stringify (łatwe do debugowania w logach)
        normalized = {k: (None if v is None else str(v)) for k, v in out.items()}
        return normalized, True, None

    validate_schema = T.StructType([
        T.StructField("normalized", T.MapType(T.StringType(), T.StringType(), valueContainsNull=True), True),
        T.StructField("is_valid", T.BooleanType(), False),
        T.StructField("error", T.StringType(), True),
    ])

    validate_udf = F.udf(_validate_and_normalize_py, validate_schema)

    def with_validation_udf(self, df: DataFrame) -> DataFrame:
        """
        Dodaje kolumnę 'validation' zawierającą:
          - is_valid
          - error
          - normalized (Map[String,String])
        """
        return (
            df.withColumn("validation", validate_udf(F.col(output_col)))
              .withColumn("is_valid", F.col("validation.is_valid"))
              .withColumn("validation_error", F.col("validation.error"))
        )

    return type(
        name,
        (object,),
        {
            "schema": struct_schema,
            "fields": fields,
            "enforce_schema": enforce_schema,
            "with_validation_udf": with_validation_udf,
        },
    )


# ---------------------------
# 3) Demo użycia (możesz zamienić na Kafka)
# ---------------------------

def main():
    spark = (
        SparkSession.builder
        .appName("DynamicSparkProcessorDemo")
        .master("local[*]")
        .getOrCreate()
    )

    # Przykładowe rekordy JSON (jak z Kafka value)
    data = [
        ('{"user_id":"123","email":"  A@B.com  ","age":"30","country":"PL"}',),
        ('{"user_id":124,"email":"X@Y.com","age":40,"country":"PL"}',),
        ('{"user_id":"oops","email":"bad@z.com","age":"NaN","country":"PL"}',),
        ('{"user_id":125,"email":"no_country@z.com","age":22}',),
        ("{not-json}",),
    ]
    df = spark.createDataFrame(data, ["raw_json"])

    # Niestandardowe reguły (UDF path)
    def norm_email(v):
        if v is None:
            return None
        return str(v).strip().lower()

    def age_validator(v):
        if v is None:
            return "age is null"
        if v < 0 or v > 120:
            return "age out of range"
        return None

    UserProcessor = create_spark_processor(
        "UserProcessor",
        fields=[
            integer("user_id", required=True),
            string("email", required=True, coercion=norm_email),
            integer("age", required=True, validator=age_validator),
            string("country", required=True),
        ],
        input_col="raw_json",
        output_col="data",
        keep_raw=True,
    )

    proc = UserProcessor()

    # (A) schema enforcement (fast)
    enforced = proc.enforce_schema(df)
    print("\n=== After schema enforcement ===")
    enforced.select("raw_json", "data", "is_schema_ok", "schema_error").show(truncate=False)

    # (B) validation/normalization UDF (optional)
    validated = proc.with_validation_udf(enforced)
    print("\n=== After validation UDF ===")
    validated.select(
        "raw_json", "data", "is_schema_ok", "schema_error", "is_valid", "validation_error", "validation.normalized"
    ).show(truncate=False)

    # Dobre rekordy
    good = validated.filter(F.col("is_schema_ok") & F.col("is_valid")).select("data.*")
    print("\n=== Good rows ===")
    good.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
