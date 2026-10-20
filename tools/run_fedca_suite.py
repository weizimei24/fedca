"""Train + evaluate FedCA, its ablations and the baselines it must beat.

Every experiment uses config.zsre_fedca.yaml (identical data, model, seed,
rounds, optimizer, local steps, rank) and differs only in the listed flags.
Outputs: exp/zsre8_chat_eos_v2/fedca_suite/<name>/.  Finished steps are
skipped, so the script can be restarted after an interruption.

  python -X utf8 -u tools/run_fedca_suite.py --stage core
  python -X utf8 -u tools/run_fedca_suite.py --stage mechanism
  python -X utf8 -u tools/run_fedca_suite.py --only fedca,srv_eta3
  python -X utf8 -u tools/run_fedca_suite.py --summary      # table only

FedIT itself is not retrained: lambda=0 / eta=1 is bitwise FedIT (see
tests/test_fedca.py), and its 10-round result lives in exp/zsre8_chat_eos_v2/fedit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "zsre8_chat_eos_v2/fedca_suite"
CONFIG = r"config/config.zsre_fedca.yaml"

# name -> (stage, flags, what it answers)
EXPERIMENTS = {
    # ---- core comparison -------------------------------------------------
    "fedca":        ("core", {}, "full method: constraint + probe absorption (gain-selected step)"),
    "fedca_k1":     ("core", {"ca_kappa": 1}, "full method with the strict excess guard"),
    "srv_eta2":     ("core", {"ca_lambda": 0, "ca_eta_mode": "fixed", "ca_eta": 2}, "FedAvg + server lr 2"),
    "srv_eta3":     ("core", {"ca_lambda": 0, "ca_eta_mode": "fixed", "ca_eta": 3}, "FedAvg + server lr 3"),
    "fedexp":       ("core", {"ca_lambda": 0, "ca_eta_mode": "fedexp"}, "FedExP adaptive server step"),
    "probe_only":   ("core", {"ca_lambda": 0}, "step 3 without step 2"),
    "ntd_only":     ("core", {"ca_eta_mode": "fixed", "ca_eta": 1}, "step 2 without step 3 (FedNTD-style)"),
    # ---- mechanism: does the constraint raise the safe absorption step? ----
    "srv_eta4":     ("mechanism", {"ca_lambda": 0, "ca_eta_mode": "fixed", "ca_eta": 4}, "server lr 4"),
    "srv_eta6":     ("mechanism", {"ca_lambda": 0, "ca_eta_mode": "fixed", "ca_eta": 6}, "server lr 6"),
    "ntd_eta2":     ("mechanism", {"ca_eta_mode": "fixed", "ca_eta": 2}, "constraint + server lr 2"),
    "ntd_eta3":     ("mechanism", {"ca_eta_mode": "fixed", "ca_eta": 3}, "constraint + server lr 3"),
    "ntd_eta4":     ("mechanism", {"ca_eta_mode": "fixed", "ca_eta": 4}, "constraint + server lr 4"),
    "ntd_eta6":     ("mechanism", {"ca_eta_mode": "fixed", "ca_eta": 6}, "constraint + server lr 6"),
    # ---- sensitivity (report all; do not pick on test) ---------------------
    "fedca_lam0.3": ("sensitivity", {"ca_lambda": 0.3}, "lambda 0.3"),
    "fedca_lam3":   ("sensitivity", {"ca_lambda": 3}, "lambda 3"),
    "fedca_beta1":  ("sensitivity", {"ca_beta": 1}, "per-client compatibility tilt"),
}


def flags(overrides: dict) -> list[str]:
    out = []
    for key, value in overrides.items():
        out += [f"--{key}", str(value)]
    return out


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(command)}\n  log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.call([sys.executable, "-X", "utf8", "-u", *command], cwd=ROOT,
                               stdout=handle, stderr=subprocess.STDOUT)
    if code:
        raise SystemExit(f"failed ({code}); see {log}")


def adapter_dir(name: str) -> Path:
    matches = sorted((ROOT / "exp" / RUN / name / "adapter").glob("fedca_*"))
    return matches[0] if matches else ROOT / "exp" / RUN / name / "adapter" / "fedca_zsre_Qwen3-1.7B_8c_1E_lr0.0001"


def summarize() -> None:
    rows = []
    fedit = ROOT / "exp/zsre8_chat_eos_v2/fedit/evaluation/zsre/metrics.json"
    if fedit.is_file():
        m = json.loads(fedit.read_text(encoding="utf-8"))["macro_average"]
        rows.append(("fedit (existing)", m["em"], m["f1"], m["eos_rate"], "-", "FedIT, 10 rounds"))
    for name, (_, _, note) in EXPERIMENTS.items():
        path = ROOT / "exp" / RUN / name / "evaluation/zsre/metrics.json"
        if not path.is_file():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))["macro_average"]
        log = adapter_dir(name) / "fedca_rounds.jsonl"
        etas = [json.loads(line)["eta"] for line in log.open(encoding="utf-8")] if log.is_file() else []
        rows.append((name, m["em"], m["f1"], m["eos_rate"], ",".join(f"{e:g}" for e in etas), note))
    print(f"{'experiment':<18}{'EM':>7}{'F1':>7}{'EOS%':>7}  eta per round / note")
    for name, em, f1, eos, etas, note in rows:
        print(f"{name:<18}{em:7.2f}{f1:7.2f}{eos:7.1f}  {etas}  ({note})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("core", "mechanism", "sensitivity", "all"), default="core")
    parser.add_argument("--only", default="", help="comma-separated experiment names")
    parser.add_argument("--rounds-eval", default="", help="also evaluate these saved rounds, e.g. 2,4,6,8")
    parser.add_argument("--resume-training", action="store_true",
                        help="continue a partial training run from its last saved completed round")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summarize()
        return 0
    names = [n for n in args.only.split(",") if n] or [
        n for n, (stage, _, _) in EXPERIMENTS.items() if args.stage in ("all", stage)]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")
    for name in names:
        _, overrides, _ = EXPERIMENTS[name]
        suffix = f"{RUN}/{name}"
        out = ROOT / "exp" / suffix
        logs = out / "logs"
        common = ["--config", CONFIG, "--suffix", suffix]
        round_log = adapter_dir(name) / "fedca_rounds.jsonl"
        if args.resume_training and round_log.is_file():
            # A root adapter may have been written by main.py's finally block
            # after an interruption, so inspect saved rounds even when it exists.
            from datetime import datetime
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run(["tools/resume_fedca.py", *common, *flags(overrides)], logs / f"resume_train_{stamp}.log")
        elif not (adapter_dir(name) / "lora_weights.pt").is_file():
            if round_log.exists():
                raise SystemExit(f"{name}: a previous training run stopped midway; "
                                 "retry with --resume-training to reuse the last completed round")
            run(["main.py", *common, *flags(overrides)], logs / "train.log")
        if not (out / "evaluation/zsre/metrics.json").is_file():
            run(["eval_zsre.py", *common, "--method", "fedca"], logs / "eval.log")
        if args.rounds_eval:
            run(["tools/eval_zsre_rounds.py", *common, "--rounds", args.rounds_eval, "--skip-existing"],
                logs / "eval_rounds.log")
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
