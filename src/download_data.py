"""Download a configured Databento sample after displaying its estimated cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import databento as db
import yaml


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", required=True, type=Path)
	parser.add_argument(
		"--confirm",
		action="store_true",
		help="Download after displaying the estimated cost.",
	)
	return parser.parse_args()


def load_config(path: Path) -> dict:
	with path.open(encoding="utf-8") as config_file:
		config = yaml.safe_load(config_file)

	if not isinstance(config, dict):
		raise ValueError("The config must contain a YAML mapping.")

	required = {"dataset", "schema", "symbols", "stype_in", "start", "end", "output"}
	missing = required.difference(config)
	if missing:
		raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
	if not config["symbols"]:
		raise ValueError("The config must contain at least one symbol.")
	return config


def sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as raw_file:
		for chunk in iter(lambda: raw_file.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def main() -> int:
	if sys.version_info < (3, 10):
		raise RuntimeError("Use Python 3.10 or newer; the project baseline is Python 3.11.")

	args = parse_args()
	config = load_config(args.config)
	api_key = os.environ.get("DATABENTO_API_KEY")
	if not api_key:
		raise RuntimeError("DATABENTO_API_KEY is not set in the terminal environment.")

	client = db.Historical(api_key)
	symbols = config["symbols"]
	cost = client.metadata.get_cost(
		dataset=config["dataset"],
		start=config["start"],
		end=config["end"],
		symbols=symbols,
		schema=config["schema"],
		stype_in=config["stype_in"],
	)
	print(f"Estimated cost: ${cost:.2f}")
	if not args.confirm:
		print("No data downloaded. Re-run with --confirm to proceed.")
		return 0

	output_path = Path(config["output"])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	store = client.timeseries.get_range(
		dataset=config["dataset"],
		start=config["start"],
		end=config["end"],
		symbols=symbols,
		schema=config["schema"],
		stype_in=config["stype_in"],
		path=output_path,
	)
	del store

	manifest = {
		"config": str(args.config),
		"query": config,
		"downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
		"file": str(output_path),
		"sha256": sha256(output_path),
		"python": platform.python_version(),
	}
	manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
	print(f"Saved raw data to {output_path}")
	print(f"Saved manifest to {manifest_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
