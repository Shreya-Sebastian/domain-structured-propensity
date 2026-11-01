"""Read and validate reasoning_model.yaml."""
from dataclasses import dataclass
from pathlib import Path

import yaml

REASONING_PATH = Path("reasoning_model.yaml")


@dataclass
class ReasoningModel:
    factors: dict          # factor id -> spec from the yaml
    feature_labels: dict
    sources: dict
    flag_features: tuple = ()

    @property
    def directions(self) -> dict:
        """feature -> 'increasing' | 'decreasing', over all factors."""
        return {f: d for spec in self.factors.values() for f, d in spec.get("directions", {}).items()}

    def monotone_vector(self, columns) -> list:
        """LightGBM monotone_constraints: +1, -1 or 0 per column."""
        sign = {"increasing": 1, "decreasing": -1}
        return [sign.get(self.directions.get(c), 0) for c in columns]

    def restrict(self, columns) -> "ReasoningModel":
        """The same reasoning model over a subset of columns; factors left
        with no features are dropped. Used by robustness checks."""
        cols = set(columns)
        factors = {}
        for fid, spec in self.factors.items():
            feats = [f for f in spec["features"] if f in cols]
            if feats:
                new = dict(spec)
                new["features"] = feats
                new["directions"] = {f: d for f, d in spec.get("directions", {}).items() if f in cols}
                factors[fid] = new
        return ReasoningModel(factors, self.feature_labels, self.sources, self.flag_features)

    def factor_pairs(self, columns) -> list:
        """All feature pairs inside each factor, for a factor-grouped EBM."""
        pairs = []
        for spec in self.factors.values():
            fs = [f for f in spec["features"] if f in columns]
            pairs += [(a, b) for i, a in enumerate(fs) for b in fs[i + 1:]]
        return pairs

    @property
    def feature_to_factor(self) -> dict:
        return {f: fid for fid, spec in self.factors.items() for f in spec["features"]}

    def interaction_groups(self, columns) -> list:
        """One list of column indices per factor, for LightGBM."""
        index = {c: i for i, c in enumerate(columns)}
        return [[index[f] for f in spec["features"]] for spec in self.factors.values()]

    def validate(self, columns) -> None:
        columns = list(columns)
        seen = {}
        for fid, spec in self.factors.items():
            for f in spec["features"]:
                if f in seen:
                    raise ValueError(f"{f} is in both {seen[f]} and {fid}")
                seen[f] = fid
            for f, direction in spec.get("directions", {}).items():
                if f not in spec["features"]:
                    raise ValueError(f"direction for {f} outside its factor {fid}")
                if direction not in ("increasing", "decreasing"):
                    raise ValueError(f"bad direction {direction} for {f}")
                if not spec.get("sources"):
                    raise ValueError(f"direction for {f} in {fid} has no source")
            for s in spec.get("sources", []):
                if s not in self.sources:
                    raise ValueError(f"unknown source key {s} in {fid}")
        missing = set(columns) - set(seen)
        extra = set(seen) - set(columns)
        if missing or extra:
            raise ValueError(f"features without a factor: {sorted(missing)}; "
                             f"factor features not in data: {sorted(extra)}")


def load(path: Path = REASONING_PATH) -> ReasoningModel:
    raw = yaml.safe_load(path.read_text())
    return ReasoningModel(raw["factors"], raw["feature_labels"], raw["sources"],
                          tuple(raw.get("flag_features", ())))
