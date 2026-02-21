# dynamic_processor_demo.py
from typing import Any, Callable, Dict, Iterable, List, Tuple
import unittest


def create_data_processor(
    name: str,
    schema: Dict[str, type],
    *,
    coercions: Dict[str, Callable[[Any], Any]] | None = None,
):
    """
    Metaprogramowanie: w runtime tworzymy klasę procesora danych na podstawie schematu.
    - schema: mapa {pole: typ}
    - coercions: opcjonalne funkcje koercji per pole (np. normalizacja email)
    """
    coercions = coercions or {}

    def validate_one(self, record: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for field, field_type in self.schema.items():
            if field not in record:
                raise ValueError(f"Missing field: {field}")

            v = record[field]
            try:
                # Najpierw koercja niestandardowa, potem typ domyślny
                if field in self.coercions:
                    v = self.coercions[field](v)
                elif not isinstance(v, field_type):
                    v = field_type(v)
            except Exception as e:
                raise TypeError(
                    f"Field '{field}' cannot be coerced to {field_type.__name__}: {v!r}"
                ) from e

            out[field] = v
        return out

    def validate_batch(
        self, records: Iterable[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Exception]]]:
        ok: List[Dict[str, Any]] = []
        bad: List[Tuple[int, Exception]] = []
        for i, r in enumerate(records):
            try:
                ok.append(self.validate_one(r))
            except Exception as e:
                bad.append((i, e))
        return ok, bad

    # Tu dzieje się „meta”: tworzymy klasę w runtime
    return type(
        name,
        (object,),
        {
            "schema": dict(schema),
            "coercions": dict(coercions),
            "validate_one": validate_one,
            "validate_batch": validate_batch,
        },
    )


# ====== DEMO: schemat + wygenerowana klasa ======
schema_v1 = {
    "user_id": int,
    "email": str,
    "age": int,
    "country": str,
}

UserProcessorV1 = create_data_processor(
    "UserProcessorV1",
    schema_v1,
    coercions={"email": lambda s: str(s).strip().lower()},
)

records = [
    {"user_id": "123", "email": "  A@B.com ", "age": "30", "country": "PL"},
    {"user_id": 124, "email": "X@Y.com", "age": 40, "country": "PL"},
    {"user_id": "oops", "email": "bad@z.com", "age": "NaN", "country": "PL"},  # fail: user_id
    {"user_id": 125, "email": "no_country@z.com", "age": 22},  # fail: missing country
]


def run_demo():
    proc = UserProcessorV1()
    ok, bad = proc.validate_batch(records)

    print("OK records:")
    for r in ok:
        print(" ", r)

    print("\nBAD records:")
    for idx, err in bad:
        print(f"  idx={idx} error={err}")


# ====== TESTY ======
class TestDynamicProcessor(unittest.TestCase):
    def setUp(self):
        self.proc = UserProcessorV1()

    def test_ok_records(self):
        ok, bad = self.proc.validate_batch(records[:2])
        self.assertEqual(len(ok), 2)
        self.assertEqual(bad, [])
        self.assertEqual(ok[0]["email"], "a@b.com")
        self.assertEqual(ok[0]["user_id"], 123)

    def test_bad_records(self):
        ok, bad = self.proc.validate_batch(records)
        self.assertEqual(len(ok), 2)
        self.assertEqual(len(bad), 2)
        self.assertIn("user_id", str(bad[0][1]))
        self.assertIn("Missing field", str(bad[1][1]))


if __name__ == "__main__":
    run_demo()
    print("\nRunning tests...\n")
    unittest.main(verbosity=2)
