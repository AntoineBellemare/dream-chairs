"""Compose a corpus of short, form-evocative dream texts for the scale demo.

SYNTHETIC PLACEHOLDER DATA. Deterministically combines curated, culturally-neutral
form archetypes with settings/actions to produce N varied dream phrases spanning many
geometries (round, tall, branching, flat, spiky, hollow, layered, flowing, creatures)
so that clustering has real structure to find. Replace the output with the real
Cosmologyscape/Oneiris dreams when ready — downstream scripts only need
{"dreams":[{"id","text"},...]}.

  python scripts/make_dreams.py --n 100
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# form archetypes grouped by gross geometry (the structure we want clusters to recover)
SUBJECTS = [
    # round / spherical
    "a smooth round boulder", "a sphere of clouded glass", "an egg on the sand",
    "a full moon low on the horizon", "a single pearl", "a soap bubble",
    # tall / vertical
    "a tall stone spire", "a thin column of smoke", "a tall spiral of light",
    "a standing stone", "a slender pillar of ice", "a totem of stacked masks",
    # branching
    "a bare winter tree", "a tangle of roots", "a nest of branches",
    "a branch of white coral", "a pair of antlers", "a forked river of light",
    # flat / planar
    "a wide flat leaf", "a slab of grey stone", "a sheet of cracked ice",
    "a folded cloth", "a single outstretched wing",
    # spiky / angular
    "a sharp mountain peak", "a jagged quartz crystal", "a shard of broken glass",
    "a five-pointed star", "a spined cactus",
    # hollow / shell
    "an empty seashell", "a shallow stone bowl", "a hollow log",
    "a carved wooden mask", "a cracked open geode",
    # layered / stacked
    "a balanced stack of stones", "a spiral staircase", "terraced hillsides",
    "a stack of thin slabs",
    # creatures
    "a hovering dragonfly", "a resting turtle", "a bird with open wings",
    "a coiled snake", "a moth at rest", "a leaping deer", "a curled fish",
    "a spider in its web", "a diving whale",
    # flowing
    "a breaking wave", "a ribbon of water", "a single flame",
    "a wisp of drifting smoke", "a billowing sail",
]

ACTIONS = [
    "hovering", "slowly turning", "rising", "dissolving", "drifting",
    "resting", "unfolding", "twisting", "floating", "breaking apart",
    "glowing softly", "held still", "tumbling", "stretching upward",
]

SETTINGS = [
    "at dusk", "over still water", "in a quiet field", "in thick fog",
    "on dark sand", "under a grey sky", "in an empty room", "on a frozen lake",
    "in a forest clearing", "against the night sky", "in a wide desert",
    "at the edge of a cliff", "in a pool of light", "among tall grass",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "dreams" / "dream_dataset.json"))
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    seen, dreams = set(), []
    tries = 0
    while len(dreams) < args.n and tries < args.n * 50:
        tries += 1
        subj = rng.choice(SUBJECTS)
        text = f"{subj}, {rng.choice(ACTIONS)} {rng.choice(SETTINGS)}"
        text = text[0].upper() + text[1:]
        if text in seen:
            continue
        seen.add(text)
        # keep the leading archetype noun as a coarse tag for later coloring/labels
        tag = subj.split()[-1]
        dreams.append({"id": f"D{len(dreams):03d}", "text": text, "subject": subj, "tag": tag})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"_note": "SYNTHETIC placeholder dreams for the scale demo; replace with real Cosmologyscape/Oneiris dreams.",
                   "dreams": dreams}, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(dreams)} dreams -> {args.out}")
    for d in dreams[:6]:
        print(f"  {d['id']}: {d['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
