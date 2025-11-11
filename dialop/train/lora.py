"""Entry-point helpers for LoRA fine-tuning data preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dialop.train.data import export_matching_turns, load_matching_data


def main(output: Path) -> None:
	data = load_matching_data()
	print(f"Loaded {len(data)} human-human matching games.")
	n_examples = export_matching_turns(data, output)
	print(f"Wrote {n_examples} turn-level examples to {output}.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Prepare LoRA fine-tuning data")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path.cwd() / "matching_turns.jsonl",
		help="Destination JSONL file for per-turn training examples.",
	)
	args = parser.parse_args()
	main(args.output)