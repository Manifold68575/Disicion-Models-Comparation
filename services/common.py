import hashlib
import json
from pathlib import Path

from decisionbench.artifacts import file_hash, write_json
from decisionbench.datasets import load_catalog, load_samples


def catalog_signature(catalog):
    return hashlib.sha256(
        json.dumps(catalog.labels, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def training_inputs(train, catalog, validation=None):
    catalog_data = load_catalog(catalog)
    samples = load_samples(train, [catalog_data])
    if validation is not None:
        valid = load_samples(validation, [catalog_data])
        if {s.sample_id for s in samples} & {s.sample_id for s in valid}:
            raise ValueError("training and validation sample IDs overlap")
        if {s.text for s in samples} & {s.text for s in valid}:
            raise ValueError("training and validation texts overlap")
    else:
        valid = []
    return catalog_data, samples, valid


def save_metadata(out: Path, catalog, train: Path, details: dict):
    weights = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    digest = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()
    write_json(
        out / "metadata.json",
        {
            **details,
            "model_version": f"{details['kind']}:{digest[:16]}",
            "catalog_signature": catalog_signature(catalog),
            "catalog_version": catalog.version,
            "train_sha256": file_hash(train),
            "artifact_files": weights,
            "labels": list(catalog.labels),
        },
    )
    write_json(
        out / "catalog.json",
        {
            "version": catalog.version,
            "labels": catalog.labels,
            "reject_label": catalog.reject_label,
        },
    )
