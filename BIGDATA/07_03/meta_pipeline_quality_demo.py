from __future__ import annotations

import inspect
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable


# =========================================================
# 1. PROTOKOŁY: kontrakty zachowania
# =========================================================

@runtime_checkable
class SupportsValidate(Protocol):
    def validate(self, value: Any) -> tuple[bool, Any, str | None]:
        ...


@runtime_checkable
class SupportsSerialize(Protocol):
    def serialize(self) -> dict[str, Any]:
        ...


# =========================================================
# 2. POLA SCHEMATU
# =========================================================

class Field(ABC):
    """
    Bazowa klasa pola schematu.
    Każde pole potrafi:
    - sprawdzić wartość
    - ewentualnie ją skonwertować
    - zwrócić błąd walidacji
    """

    def __init__(self, *, required: bool = True, default: Any = None):
        self.required = required
        self.default = default
        self.name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @abstractmethod
    def validate(self, value: Any) -> tuple[bool, Any, str | None]:
        raise NotImplementedError


class StringField(Field):
    def __init__(
        self,
        *,
        required: bool = True,
        default: Any = None,
        min_length: int | None = None,
        pattern: str | None = None,
    ):
        super().__init__(required=required, default=default)
        self.min_length = min_length
        self.pattern = re.compile(pattern) if pattern else None

    def validate(self, value: Any) -> tuple[bool, Any, str | None]:
        if value is None:
            if self.required:
                return False, None, f"{self.name}: missing required value"
            return True, self.default, None

        coerced = str(value).strip()

        if self.min_length is not None and len(coerced) < self.min_length:
            return False, coerced, f"{self.name}: too short"

        if self.pattern and not self.pattern.match(coerced):
            return False, coerced, f"{self.name}: invalid format"

        return True, coerced, None


class IntegerField(Field):
    def __init__(
        self,
        *,
        required: bool = True,
        default: Any = None,
        min_value: int | None = None,
        max_value: int | None = None,
    ):
        super().__init__(required=required, default=default)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> tuple[bool, Any, str | None]:
        if value is None or value == "":
            if self.required:
                return False, None, f"{self.name}: missing required value"
            return True, self.default, None

        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return False, value, f"{self.name}: not an integer"

        if self.min_value is not None and coerced < self.min_value:
            return False, coerced, f"{self.name}: below minimum {self.min_value}"

        if self.max_value is not None and coerced > self.max_value:
            return False, coerced, f"{self.name}: above maximum {self.max_value}"

        return True, coerced, None


class FloatField(Field):
    def __init__(
        self,
        *,
        required: bool = True,
        default: Any = None,
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        super().__init__(required=required, default=default)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> tuple[bool, Any, str | None]:
        if value is None or value == "":
            if self.required:
                return False, None, f"{self.name}: missing required value"
            return True, self.default, None

        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return False, value, f"{self.name}: not a float"

        if self.min_value is not None and coerced < self.min_value:
            return False, coerced, f"{self.name}: below minimum {self.min_value}"

        if self.max_value is not None and coerced > self.max_value:
            return False, coerced, f"{self.name}: above maximum {self.max_value}"

        return True, coerced, None


class ChoiceField(Field):
    def __init__(self, choices: set[str], *, required: bool = True, default: Any = None):
        super().__init__(required=required, default=default)
        self.choices = choices

    def validate(self, value: Any) -> tuple[bool, Any, str | None]:
        if value is None:
            if self.required:
                return False, None, f"{self.name}: missing required value"
            return True, self.default, None

        coerced = str(value).strip()
        if coerced not in self.choices:
            return False, coerced, f"{self.name}: must be one of {sorted(self.choices)}"

        return True, coerced, None


# =========================================================
# 3. METAKLASA: zbieranie pól schematu
# =========================================================

class SchemaMeta(type):
    """
    Metaklasa automatycznie wykrywa pola Field w klasie
    i buduje z nich mapę __fields__.
    """

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        inherited_fields: dict[str, Field] = {}
        for base in bases:
            inherited_fields.update(getattr(base, "__fields__", {}))

        own_fields = {
            key: value
            for key, value in namespace.items()
            if isinstance(value, Field)
        }

        cls = super().__new__(mcls, name, bases, namespace)
        cls.__fields__ = {**inherited_fields, **own_fields}
        return cls


# =========================================================
# 4. SCHEMA RECORD + INTROSPEKCJA
# =========================================================

@dataclass
class ValidationErrorInfo:
    field_name: str
    message: str
    bad_value: Any


@dataclass
class ValidationResult:
    ok: bool
    cleaned: dict[str, Any]
    errors: list[ValidationErrorInfo] = field(default_factory=list)

    def serialize(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "cleaned": self.cleaned,
            "errors": [
                {
                    "field_name": e.field_name,
                    "message": e.message,
                    "bad_value": e.bad_value,
                }
                for e in self.errors
            ],
        }


class RecordSchema(metaclass=SchemaMeta):
    __fields__: dict[str, Field]

    @classmethod
    def validate_record(cls, raw: dict[str, Any]) -> ValidationResult:
        cleaned: dict[str, Any] = {}
        errors: list[ValidationErrorInfo] = []

        for field_name, field_def in cls.__fields__.items():
            raw_value = raw.get(field_name)
            ok, value, err = field_def.validate(raw_value)
            if ok:
                cleaned[field_name] = value
            else:
                errors.append(
                    ValidationErrorInfo(
                        field_name=field_name,
                        message=err or "unknown error",
                        bad_value=raw_value,
                    )
                )

        return ValidationResult(ok=len(errors) == 0, cleaned=cleaned, errors=errors)

    @classmethod
    def schema_description(cls) -> dict[str, Any]:
        """
        Introspekcja schematu: zwróć informacje o polach.
        """
        desc: dict[str, Any] = {}

        for name, field_def in cls.__fields__.items():
            desc[name] = {
                "type": field_def.__class__.__name__,
                "required": field_def.required,
                "attributes": {
                    key: value
                    for key, value in vars(field_def).items()
                    if key not in {"name", "required", "default"}
                },
            }

        return desc


class UserEventSchema(RecordSchema):
    user_id = StringField(required=True, min_length=3, pattern=r"^U\d+$")
    age = IntegerField(required=True, min_value=18, max_value=100)
    country = StringField(required=True, min_length=2)
    salary = FloatField(required=True, min_value=0.0)
    segment = ChoiceField({"A", "B", "C"}, required=True)


# =========================================================
# 5. ABSTRAKCJE PIPELINE
# =========================================================

class PipelineStage(ABC):
    @abstractmethod
    def run(self, data: Iterable[Any]) -> Iterator[Any]:
        raise NotImplementedError


class SourceStage(PipelineStage, ABC):
    pass


class TransformStage(PipelineStage, ABC):
    pass


class SinkStage(PipelineStage, ABC):
    pass


# =========================================================
# 6. ETAPY KONKRETNE
# =========================================================

class JSONLSource(SourceStage):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def run(self, data: Iterable[Any] = ()) -> Iterator[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    payload["_line_no"] = line_no
                    yield payload
                except json.JSONDecodeError:
                    yield {
                        "_line_no": line_no,
                        "_corrupt_record": line,
                    }


class SchemaValidator(TransformStage):
    def __init__(self, schema: type[RecordSchema]):
        self.schema = schema

    def run(self, data: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in data:
            if "_corrupt_record" in row:
                yield {
                    "status": "bad",
                    "raw": row,
                    "cleaned": {},
                    "errors": [
                        {
                            "field_name": "__record__",
                            "message": "invalid JSON",
                            "bad_value": row["_corrupt_record"],
                        }
                    ],
                }
                continue

            result = self.schema.validate_record(row)
            yield {
                "status": "good" if result.ok else "bad",
                "raw": row,
                "cleaned": result.cleaned,
                "errors": result.serialize()["errors"],
            }


class EnrichmentStage(TransformStage):
    """
    Dodatkowa transformacja pokazująca, że pipeline może
    budować nową semantykę po walidacji.
    """

    def run(self, data: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in data:
            if row["status"] == "good":
                salary = row["cleaned"]["salary"]
                row["cleaned"]["salary_band"] = (
                    "low" if salary < 3000
                    else "mid" if salary < 7000
                    else "high"
                )
            yield row


class SplitSink(SinkStage):
    def __init__(self):
        self.good_records: list[dict[str, Any]] = []
        self.bad_records: list[dict[str, Any]] = []

    def run(self, data: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in data:
            if row["status"] == "good":
                self.good_records.append(row)
            else:
                self.bad_records.append(row)
            yield row


class QualityReportSink(SinkStage):
    def __init__(self):
        self.total = 0
        self.good = 0
        self.bad = 0
        self.error_counts: dict[str, int] = {}

    def run(self, data: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in data:
            self.total += 1
            if row["status"] == "good":
                self.good += 1
            else:
                self.bad += 1
                for err in row["errors"]:
                    key = f"{err['field_name']} -> {err['message']}"
                    self.error_counts[key] = self.error_counts.get(key, 0) + 1
            yield row

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "good": self.good,
            "bad": self.bad,
            "quality_ratio": round(self.good / self.total, 4) if self.total else 0.0,
            "top_errors": dict(
                sorted(self.error_counts.items(), key=lambda kv: kv[1], reverse=True)
            ),
        }


# =========================================================
# 7. ORKIESTRATOR PIPELINE + INTROSPEKCJA
# =========================================================

class DataPipeline:
    def __init__(self, *stages: PipelineStage):
        self.stages = stages

    def execute(self) -> list[Any]:
        stream: Iterable[Any] = ()
        for i, stage in enumerate(self.stages):
            if i == 0 and isinstance(stage, SourceStage):
                stream = stage.run(())
            else:
                stream = stage.run(stream)
        return list(stream)

    def describe(self) -> None:
        print("\n=== PIPELINE INTROSPECTION ===")
        for idx, stage in enumerate(self.stages, start=1):
            print(f"\n[{idx}] {stage.__class__.__name__}")
            print(f"  MRO: {[cls.__name__ for cls in inspect.getmro(stage.__class__)]}")

            sig = inspect.signature(stage.run)
            print(f"  run signature: {stage.__class__.__name__}.run{sig}")

            methods = [
                name for name, member in inspect.getmembers(stage, predicate=inspect.ismethod)
                if not name.startswith("_")
            ]
            print(f"  public methods: {methods}")

        print("\n=== SCHEMA INTROSPECTION ===")
        print(json.dumps(UserEventSchema.schema_description(), indent=2, ensure_ascii=False))


# =========================================================
# 8. DANE DEMO
# =========================================================

def create_demo_file(path: str | Path) -> Path:
    path = Path(path)
    rows = [
        {"user_id": "U100", "age": 28, "country": "PL", "salary": 4200, "segment": "A"},
        {"user_id": "U101", "age": "35", "country": "DE", "salary": "7100.5", "segment": "B"},
        {"user_id": "broken", "age": 16, "country": "PL", "salary": 2000, "segment": "A"},
        {"user_id": "U103", "age": "abc", "country": "FR", "salary": 3000, "segment": "X"},
        {"user_id": "U104", "age": 42, "country": "", "salary": -100, "segment": "C"},
        {"user_id": "U105", "age": 51, "country": "ES", "salary": 8300, "segment": "B"},
    ]

    corrupt_line = '{"user_id": "U999", "age": 30, "country": "PL", "salary": 5000, "segment": "A"'

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(corrupt_line + "\n")

    return path


# =========================================================
# 9. MAIN DEMO
# =========================================================

def main() -> None:
    input_path = create_demo_file("demo_dirty_users.jsonl")

    source = JSONLSource(input_path)
    validator = SchemaValidator(UserEventSchema)
    enrich = EnrichmentStage()
    splitter = SplitSink()
    reporter = QualityReportSink()

    pipeline = DataPipeline(
        source,
        validator,
        enrich,
        splitter,
        reporter,
    )

    pipeline.describe()
    _ = pipeline.execute()

    print("\n=== GOOD RECORDS ===")
    for row in splitter.good_records:
        print(json.dumps(row, indent=2, ensure_ascii=False))

    print("\n=== BAD RECORDS ===")
    for row in splitter.bad_records:
        print(json.dumps(row, indent=2, ensure_ascii=False))

    print("\n=== QUALITY REPORT ===")
    print(json.dumps(reporter.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
