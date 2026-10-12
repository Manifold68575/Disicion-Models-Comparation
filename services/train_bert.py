import argparse
from pathlib import Path

from decisionbench.artifacts import file_hash
from services.common import save_metadata, training_inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT; validation split required")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--revision", required=True, help="Freeze the Hugging Face model commit")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    catalog, train, valid = training_inputs(args.train, args.catalog, args.validation)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output must be empty; use a new version directory")
    labels = list(catalog.labels)
    label_to_id = {label: i for i, label in enumerate(labels)}
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        revision=args.revision,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id=label_to_id,
    )

    def encode(samples):
        dataset = Dataset.from_dict(
            {"text": [s.text for s in samples], "labels": [label_to_id[s.label] for s in samples]}
        )
        return dataset.map(
            lambda row: tokenizer(row["text"], truncation=True, max_length=args.max_length),
            batched=True,
            remove_columns=["text"],
        )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=2e-5,
            seed=args.seed,
            data_seed=args.seed,
            full_determinism=True,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=encode(train),
        eval_dataset=encode(valid),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    validation_metrics = trainer.evaluate()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(args.out)
    save_metadata(
        args.out,
        catalog,
        args.train,
        {
            "kind": "bert",
            "seed": args.seed,
            "base_model": args.base_model,
            "revision": args.revision,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "validation_sha256": file_hash(args.validation),
            "validation_metrics": validation_metrics,
        },
    )
    print(f"Saved BERT: {args.out}")


if __name__ == "__main__":
    main()
