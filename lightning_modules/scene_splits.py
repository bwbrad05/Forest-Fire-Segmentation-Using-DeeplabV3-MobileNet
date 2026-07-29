"""
Scene-aware 5-fold splitting for the Indonesian burned-area dataset.

Pure Python (no torch/torchvision), so the split script can import it without
pulling in the training stack.

Why this exists
---------------
Tile filenames follow ``L8_<pathrow>_<date>_<tile>.tif``
(e.g. ``L8_117060_240919_003``). Every tile that shares ``<pathrow>_<date>``
comes from the SAME Landsat acquisition — same fire, same day, same atmosphere.
If those tiles are scattered across folds, a model gets tested on a tile right
next to one it trained on, which measures memorisation rather than
generalisation and inflates IoU/F1. Grouping whole scenes into a single fold
removes that spatial leakage.
"""

from collections import defaultdict
from typing import Dict, Iterable, List


def scene_key(file_id: str) -> str:
    """
    Return the scene identifier shared by all tiles of one acquisition.

    Drops the trailing tile number:
        'L8_117060_240919_003' -> 'L8_117060_240919'

    A name with no separable tile suffix is treated as its own scene.
    """
    parts = file_id.split("_")
    if len(parts) <= 1:
        return file_id
    return "_".join(parts[:-1])


def assign_scene_folds(file_ids: Iterable[str], n_folds: int = 5) -> Dict[str, int]:
    """
    Map each file id to a fold so that whole scenes stay together.

    Folds are balanced by TILE count with a deterministic greedy pass: scenes
    are processed largest-first and each is placed in the fold that currently
    holds the fewest tiles (ties broken by fold index). No RNG is used, so the
    result is identical on every run.

    Parameters
    ----------
    file_ids : iterable of str
        Tile stems (without extension), e.g. ``L8_117060_240919_003``.
    n_folds : int
        Number of folds (5 for this dataset).

    Returns
    -------
    dict
        ``{file_id: fold_index}`` for every input id.
    """
    scenes: Dict[str, List[str]] = defaultdict(list)
    for fid in file_ids:
        scenes[scene_key(fid)].append(fid)

    # Largest scenes first; tie-break by scene key so the order is stable.
    ordered = sorted(scenes.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    fold_load = [0] * n_folds
    mapping: Dict[str, int] = {}
    for key, tiles in ordered:
        target = min(range(n_folds), key=lambda i: (fold_load[i], i))
        for fid in tiles:
            mapping[fid] = target
        fold_load[target] += len(tiles)
    return mapping


def count_cross_fold_scenes(mapping: Dict[str, int]) -> int:
    """
    Number of scenes whose tiles ended up in more than one fold.

    Zero means there is no scene-level leakage. Use this to sanity-check any
    split (a per-tile random split will report a large number here).
    """
    scene_folds: Dict[str, set] = defaultdict(set)
    for fid, fold in mapping.items():
        scene_folds[scene_key(fid)].add(fold)
    return sum(1 for folds in scene_folds.values() if len(folds) > 1)


def fold_summary(mapping: Dict[str, int], n_folds: int = 5) -> List[Dict[str, int]]:
    """Return per-fold counts of tiles and distinct scenes, for reporting."""
    tiles_per_fold = [0] * n_folds
    scenes_per_fold: List[set] = [set() for _ in range(n_folds)]
    for fid, fold in mapping.items():
        tiles_per_fold[fold] += 1
        scenes_per_fold[fold].add(scene_key(fid))
    return [
        {"fold": i, "tiles": tiles_per_fold[i], "scenes": len(scenes_per_fold[i])}
        for i in range(n_folds)
    ]
