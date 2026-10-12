import argparse
from pathlib import Path

from services.common import save_metadata, training_inputs


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + linear SVM on training split only")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from joblib import dump
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    catalog, samples, _ = training_inputs(args.train, args.catalog)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output must be empty; use a new version directory")
    args.out.mkdir(parents=True, exist_ok=True)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        LinearSVC(random_state=args.seed, dual="auto"),
    )
    model.fit([s.text for s in samples], [s.label for s in samples])
    dump(model, args.out / "model.joblib")
    save_metadata(args.out, catalog, args.train, {"kind": "tfidf-svm", "seed": args.seed})
    print(f"Saved classifier: {args.out}")


if __name__ == "__main__":
    main()
