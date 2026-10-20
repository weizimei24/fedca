"""Summarize the zsRE factual-knowledge consolidation comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KNOWN_METHODS = ("base", "local", "centralized", "fedit", "fedrotlora", "rolora", "flora", "fedexlora", "ffalora", "flexlora", "ilora", "fedfh", "fedca", "fedsvd", "fedmomentum")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="zsre8_stage1", help="directory below exp/")
    parser.add_argument("--metric", choices=("em", "f1", "containment"), default="em")
    parser.add_argument("--centralized-dir", default="centralized",
                        help="output directory holding the Centralized evaluation")
    parser.add_argument("--output", default=None,
                        help="optional summary CSV name under the run directory")
    parser.add_argument(
        "--methods", default="base,local,fedit,fedrotlora",
        help="comma-separated evaluated methods; use base,local,centralized,fedit for the original comparison",
    )
    args = parser.parse_args()
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if len(methods) != len(set(methods)) or any(method not in KNOWN_METHODS for method in methods):
        raise ValueError(f"--methods must be unique values from: {', '.join(KNOWN_METHODS)}")
    if "base" not in methods or "local" not in methods:
        raise ValueError("--methods must include base and local to compute LocalGain")
    root = Path("exp") / args.run
    payloads = {}
    for method in methods:
        directory = args.centralized_dir if method == 'centralized' else method
        if Path(directory).name != directory or directory in ('.', '..'):
            raise ValueError('--centralized-dir must be a single directory name')
        path = root / directory / "evaluation" / "zsre" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; evaluate {method} first with eval_zsre.py")
        payloads[method] = json.loads(path.read_text(encoding="utf-8"))
    client_ids = sorted(payloads["base"]["per_client"], key=int)
    for method, payload in payloads.items():
        if payload.get('generation') != payloads['base'].get('generation'):
            raise ValueError(f'{method}: generation protocol differs from Base')
        if set(payload['per_client']) != set(client_ids) or any(
            payload['per_client'][i]['num_samples'] != payloads['base']['per_client'][i]['num_samples']
            for i in client_ids
        ):
            raise ValueError(f'{method}: evaluation sample counts differ from Base')
    rows = []
    for client_id in client_ids:
        values = {method: payloads[method]["per_client"][client_id][args.metric] for method in methods}
        row = {"client": client_id, **values, "local_gain": values["local"] - values["base"]}
        if "fedit" in methods:
            row["fedit_consolidation_gap"] = values["local"] - values["fedit"]
        if "fedrotlora" in methods:
            row["fedrotlora_consolidation_gap"] = values["local"] - values["fedrotlora"]
        if "rolora" in methods:
            row["rolora_consolidation_gap"] = values["local"] - values["rolora"]
        if "flora" in methods:
            row["flora_consolidation_gap"] = values["local"] - values["flora"]
        for method in ("fedexlora", "ffalora", "flexlora", "ilora", "fedfh", "fedca", "fedsvd", "fedmomentum"):
            if method in methods:
                row[f"{method}_consolidation_gap"] = values["local"] - values[method]
        rows.append(row)
    macro_values = {method: payloads[method]["macro_average"][args.metric] for method in methods}
    macro = {"client": "macro_average", **macro_values, "local_gain": macro_values["local"] - macro_values["base"]}
    if "fedit" in methods:
        macro["fedit_consolidation_gap"] = macro_values["local"] - macro_values["fedit"]
    if "fedrotlora" in methods:
        macro["fedrotlora_consolidation_gap"] = macro_values["local"] - macro_values["fedrotlora"]
    if "rolora" in methods:
        macro["rolora_consolidation_gap"] = macro_values["local"] - macro_values["rolora"]
    if "flora" in methods:
        macro["flora_consolidation_gap"] = macro_values["local"] - macro_values["flora"]
    for method in ("fedexlora", "ffalora", "flexlora", "ilora", "fedfh", "fedca", "fedsvd", "fedmomentum"):
        if method in methods:
            macro[f"{method}_consolidation_gap"] = macro_values["local"] - macro_values[method]
    rows.append(macro)
    filename = args.output or f"stage1_summary_{args.metric}.csv"
    if Path(filename).name != filename or not filename.endswith('.csv'):
        raise ValueError('--output must be a CSV filename, not a path')
    path = root / filename
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["client", *methods, "local_gain"]
        if "fedit" in methods:
            fieldnames.append("fedit_consolidation_gap")
        if "fedrotlora" in methods:
            fieldnames.append("fedrotlora_consolidation_gap")
        if "rolora" in methods:
            fieldnames.append("rolora_consolidation_gap")
        if "flora" in methods:
            fieldnames.append("flora_consolidation_gap")
        for method in ("fedexlora", "ffalora", "flexlora", "ilora", "fedfh", "fedca", "fedsvd", "fedmomentum"):
            if method in methods:
                fieldnames.append(f"{method}_consolidation_gap")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\t".join(
        "LocalGain" if field == "local_gain" else field.replace("_", "").title()
        for field in fieldnames
    ))
    for row in rows:
        print("\t".join([str(row["client"])] + [f"{row[field]:.2f}" for field in fieldnames[1:]]))
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
