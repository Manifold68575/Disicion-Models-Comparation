import json
from pathlib import Path

from decisionbench.protocol import Catalog, Sample


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                    records.append(row)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON object") from exc
    return records


def load_catalog(path: Path) -> Catalog:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels")
    if not isinstance(data.get("version"), str) or not data["version"]:
        raise ValueError("catalog.version must be a nonempty string")
    if (
        not isinstance(labels, dict)
        or not labels
        or not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in labels.items())
    ):
        raise ValueError("catalog.labels must map nonempty names to descriptions")
    reject = data.get("reject_label")
    if reject is not None and reject not in labels:
        raise ValueError("reject_label must be in catalog.labels")
    return Catalog(data["version"], labels, reject)


def load_samples(path: Path, catalogs: list[Catalog]) -> list[Sample]:
    labels = set().union(*(c.labels for c in catalogs))
    samples, seen = [], set()
    for row in read_jsonl(path):
        if not all(isinstance(row.get(k), str) and row[k] for k in ("sample_id", "text", "label")):
            raise ValueError("samples require nonempty sample_id, text and label strings")
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id: {row['sample_id']}")
        if row["label"] not in labels:
            raise ValueError(f"unknown ground-truth label: {row['label']}")
        seen.add(row["sample_id"])
        samples.append(Sample(**{k: row[k] for k in ("sample_id", "text", "label")}))
    if not samples:
        raise ValueError("dataset is empty")
    return samples
