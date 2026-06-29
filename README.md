# Dream → Chairs

A computational toolkit for turning **dream reports into 3D forms and chairs** — part of
the Cosmologyscape / Oneiris project. It generates 3D shapes from dream text, places them
in a shared 3D↔text latent space, finds families of dreams, characterizes them with design
vocabularies, and links a dream's phenomenology to a chair's posture.

It is a **set of interchangeable approaches, not one fixed pipeline.** See the design-team
handout for the menu of paths: [docs/SUMMARY_FOR_STUDENTS.md](docs/SUMMARY_FOR_STUDENTS.md)
([PDF](docs/SUMMARY_FOR_STUDENTS.pdf)).

> ⚠️ **Cultural note.** `data/symbols/symbols.template.json` is **placeholder only**. Lakota
> symbols, their meanings, and any spatial qualities must be defined and reviewed by a
> Lakota Knowledge Holder before real use. The posture rubric and design-descriptor
> families are culturally neutral design instruments and safe to edit.

---

## Pipeline overview

```
dream text ─▶ 3D form (Point-E / Shap-E) ─▶ OpenShape embedding ──┐
                                          (shared 3D ↔ text space) │
            ┌───────────────────────────┬─────────────────────────┼───────────────────────┐
            ▼                            ▼                         ▼                        ▼
   classify vs symbols /        cluster the corpus        project onto posture      dream → chair
   design descriptors          (families of dreams)       axes (dream↔body)        (generate & select)
```

Each stage has alternatives (cluster on shape vs text vs posture vs symbol; posture from
text vs shape; Point-E vs Shap-E; …) — that's the point. The handout lays them out.

---

## Setup

Built and tested in a conda env `py310` on Windows + NVIDIA GPU, with
`torch 2.5.1+cu124` already working.

```bash
conda activate py310

# core deps (does not touch your torch install)
pip install -r requirements.txt

# 3D generators (from GitHub)
pip install git+https://github.com/openai/point-e.git
pip install git+https://github.com/openai/shap-e.git
pip install ipywidgets            # shap-e notebook util import

# optional extras
pip install reportlab             # PDF handout (scripts/md_to_pdf.py)
pip install anthropic             # LLM posture rubric (extract_posture.py --method llm)
```

OpenShape is **vendored** in `dream_chairs/openshape_vendor/` (Apache-2.0) with its DGL
dependency removed, so it runs on Windows with no extra build. Its support deps
(`torch-redstone`, `einops`, `huggingface_hub`) are in `requirements.txt`; the OpenAI CLIP
text encoder it uses comes in via shap-e. Model weights download on first use.

---

## Quickstart (end to end)

```bash
python scripts/make_dreams.py --n 100                              # 1. build a dream corpus
python scripts/generate_dataset.py --backend point_e --tag pointe # 2. dream -> 3D (resumable)
python scripts/cluster_dataset.py --tag pointe                    # 3. families + design signatures
python scripts/posture_map.py --tag pointe                        # 4. place forms on body axes
python scripts/extract_posture.py --method clip                   # 5. dream text -> posture
python scripts/posture_chair.py --ids D030,D076 --variants 4      # 6. dream -> chair
```

Smaller / inspection helpers:

```bash
python scripts/render_dreams.py --tag pointe --first 24    # eyeball generated forms
python scripts/smoke_test.py                               # wiring check, no model downloads
python scripts/md_to_pdf.py docs/SUMMARY_FOR_STUDENTS.md   # rebuild the handout PDF
```

A full reference for every script is in **[docs/SCRIPTS.md](docs/SCRIPTS.md)**.

---

## Repository layout

```
dream-chairs/
├── config/config.yaml            # all tunable parameters (generation, embedding, clustering)
├── data/
│   ├── dreams/                   # dream corpora (example, series, dataset)
│   ├── symbols/                  # Lakota symbol descriptors (PLACEHOLDER)
│   └── descriptors/              # visual dictionary, design families, body-posture axes
├── dream_chairs/                 # the library
│   ├── generate.py               # text -> 3D (Point-E / Shap-E)
│   ├── embed.py                  # OpenShape + CLIP multi-view embedders
│   ├── classify.py               # cosine matching vs descriptors
│   ├── cluster.py                # clustering + emergent-term description
│   ├── families.py               # position a shape within design families
│   ├── axes.py                   # bipolar posture axes + chair-parameter mapping
│   ├── pipeline.py / cli.py      # small end-to-end runner
│   └── openshape_vendor/         # vendored OpenShape encoder (DGL-free)
├── scripts/                      # runnable entry points (see docs/SCRIPTS.md)
├── docs/                         # handout (md + pdf), script guide
└── outputs/                      # generated forms, clusters, maps, chairs (gitignored)
```

---

## Scripts at a glance

| Script | Purpose |
|--------|---------|
| `make_dreams.py` | Build a synthetic dream corpus (swap in real reports) |
| `generate_dataset.py` | Generate a 3D form per dream (Point-E / Shap-E), resumable |
| `cluster_dataset.py` | Cluster forms + contrastive design-family signatures + t-SNE |
| `posture_map.py` | Project forms onto the 6 body-posture axes (+ chair params) |
| `extract_posture.py` | Dream text → posture scores (CLIP zero-shot or LLM rubric) |
| `posture_chair.py` | Dream → chair: generate K, keep the best posture match |
| `render_dreams.py` | Labeled multi-view montages of generated forms |
| `design_families.py` | Per-dream design-family positioning (small-set version) |
| `reclassify_openshape.py` | Classify saved forms against symbols via OpenShape |
| `dream_series_demo.py` | Original 8-form demo (symbol-guided gen + classify + cluster) |
| `smoke_test.py` | Wiring check on random clouds (no model downloads) |
| `md_to_pdf.py` | Render a Markdown doc to PDF |

---

## Embedding backends

- **OpenShape (default, recommended)** — native point-cloud↔text in a shared CLIP space,
  no rendering. Vendored DGL-free; `vitl14` reuses the cached OpenAI CLIP ViT-L/14 text
  encoder (no large download). Set `embedding.backend: openshape` in the config.
- **CLIP multi-view (fallback)** — renders a form to several views and encodes with
  OpenCLIP. Simpler but markedly weaker for clustering. `embedding.backend: clip_multiview`.

---

## Credits & licenses

- **OpenShape** (Liu et al., NeurIPS 2023) — point encoder, vendored under Apache-2.0.
- **Point-E**, **Shap-E** (OpenAI) — text-to-3D generators, MIT.
- **OpenCLIP / CLIP** — text/image encoders.

This repository's own code is research scaffolding for the Cosmologyscape / Oneiris
collaboration.
