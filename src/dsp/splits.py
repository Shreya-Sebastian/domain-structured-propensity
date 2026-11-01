"""Time blocks from config/splits.yaml."""
from pathlib import Path

import pandas as pd
import yaml

SPLITS_PATH = Path("config/splits.yaml")


def load_blocks(path: Path = SPLITS_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def block_masks(meta: pd.DataFrame, blocks: dict) -> dict:
    """Boolean mask per block. Raises if blocks overlap or are out of order."""
    masks = {}
    previous_end = ""
    for name, (start, end) in blocks.items():
        if start <= previous_end:
            raise ValueError(f"block {name} overlaps or is out of order")
        masks[name] = ((meta["ym"] >= start) & (meta["ym"] <= end)).to_numpy()
        previous_end = end
    return masks
