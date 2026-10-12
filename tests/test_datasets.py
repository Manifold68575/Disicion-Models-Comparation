import json

from scripts.prepare_datasets import banking, massive, stratified_subset


def test_massive_uses_native_partitions_and_locale(tmp_path):
    path = tmp_path / "en-US.jsonl"
    raw = [
        {
            "id": str(i),
            "locale": "en-US",
            "partition": partition,
            "intent": "alarm_set",
            "utt": f"alarm {i}",
        }
        for i, partition in enumerate(["train", "dev", "test"])
    ]
    path.write_text("\n".join(json.dumps(r) for r in raw))
    splits, hashes = massive(path, "en-US")
    assert set(splits) == {"train", "validation", "test"}
    assert splits["validation"][0]["sample_id"] == "massive-en-US-validation-1"
    assert len(hashes[str(path)]) == 64


def test_banking_holdout_is_deterministic_and_test_unchanged(tmp_path):
    (tmp_path / "train.csv").write_text(
        "text,category\n"
        + "\n".join(f"train-{label}-{i},{label}" for label in ("a", "b") for i in range(20))
    )
    (tmp_path / "test.csv").write_text("text,category\ntest-a,a\ntest-b,b\n")
    first, _ = banking(tmp_path, 42)
    second, _ = banking(tmp_path, 42)
    assert first == second
    assert len(first["train"]) == 36
    assert len(first["validation"]) == 4
    assert len(first["test"]) == 2
    assert not {r["sample_id"] for r in first["train"]} & {
        r["sample_id"] for r in first["validation"]
    }
    selected = stratified_subset(first["validation"], 2, 42)
    assert {r["label"] for r in selected} == {"a", "b"}
    assert selected == stratified_subset(first["validation"], 2, 42)
