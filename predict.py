"""Regenerate a submission from a portable model; no retraining required"""
import argparse
import json
from pathlib import Path
import pandas as pd
from polynomial import load_data, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text())
    frame = load_data(args.test, model["input_columns"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": predict(model, frame)}).to_csv(output, index=False)
    print(f"Saved {len(frame)} predictions to {output}")


if __name__ == "__main__":
    main()
