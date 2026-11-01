"""Run the demo and write its outputs to results/demo/.

Usage, from the repository root:
    .venv/Scripts/python scripts/run_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsp import brief, demo  # noqa: E402

OUT = Path("results/demo")


def main():
    out = demo.run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records").mkdir(exist_ok=True)

    with open(OUT / "records" / "pilot_records.jsonl", "w") as fh:
        for r in out["records"]:
            fh.write(json.dumps(r) + "\n")

    examples = demo.example_records(out["records"])
    briefs = ["# Advisor briefs: three pilot clients", "",
              "Chosen by rule: top-ranked, at the top-10% boundary, median.", ""]
    for r in examples:
        (OUT / "records" / f"{r['client_ref']}.json").write_text(json.dumps(r, indent=2))
        briefs += [brief.render(r, out["reasoning"]), ""]
    (OUT / "briefs.md").write_text("\n".join(briefs))

    summary = {
        "label": "PILOT on validation months, not a result. Test block not scored.",
        "block_counts": out["block_counts"],
        "base_rates": out["base_rates"],
        "pilot_metrics": out["pilot_metrics"],
        "checks": out["checks"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\n" + "\n".join(briefs))


if __name__ == "__main__":
    main()
