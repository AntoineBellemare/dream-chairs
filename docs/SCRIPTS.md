# Scripts guide

Short reference for every script in `scripts/`. Run from the repo root with the `py310`
conda env active. Most read parameters from `config/config.yaml`; flags override.

Outputs are namespaced by `--tag` under `outputs/dataset_<tag>/` so you can keep multiple
generators (e.g. `pointe`, `shape64`) side by side.

---

## Corpus & generation

### `make_dreams.py` — build a dream corpus
Composes N short, form-evocative synthetic dreams (placeholder; replace with real reports).
- **Key args:** `--n 100`, `--seed 7`, `--out data/dreams/dream_dataset.json`
- **Out:** `data/dreams/dream_dataset.json` (`{id, text, subject, tag}`)
- **Example:** `python scripts/make_dreams.py --n 100`

### `generate_dataset.py` — dream → 3D form (whole corpus, resumable)
Generates one form per dream with Point-E or Shap-E. Skips dreams already done.
- **Key args:** `--backend point_e|shap_e`, `--tag <name>`, `--steps 64` (Shap-E), `--limit N`
- **Out:** `outputs/dataset_<tag>/` → `pointclouds/*.npz`, `thumbs/*.png`, `meshes/*.obj`
  (Shap-E only), `contact_sheet.png`
- **Example:** `python scripts/generate_dataset.py --backend shap_e --steps 64 --tag shape64`

---

## Families & characterization

### `cluster_dataset.py` — cluster forms + name the clusters
Embeds forms with OpenShape (cached), picks `k` by silhouette, clusters, and characterizes
each cluster with **contrastive** design-family signatures, emergent descriptors, exemplar
dreams, and a t-SNE map.
- **Key args:** `--tag <name>`, `--k N` (else auto), `--kmin/--kmax`, `--no-cache`
- **Out:** `outputs/dataset_<tag>/` → `cluster_report.json/.md`, `cluster_scatter.json`,
  `shape_embeddings_openshape.npz`
- **Example:** `python scripts/cluster_dataset.py --tag pointe`

### `design_families.py` — per-dream design-family positioning
For each form, its soft position within every design family (smaller-set version of the
above; good for inspecting individual dreams).
- **Out:** `outputs/dataset_<tag>/design_profiles.json/.md`, `design_matrix.json`
- **Example:** `python scripts/design_families.py --tag pointe`

### `reclassify_openshape.py` — classify forms against symbols
Cosine-matches saved forms to the symbol set (and descriptor dictionary) in OpenShape space.
- **Key args:** `--checkpoint`, `--clip-backend openai|openclip`, `--swap-yz`
- **Out:** `outputs/dataset_<tag>/report_openshape.json/.md`
- **Example:** `python scripts/reclassify_openshape.py --tag pointe` *(note: this one reads
  `outputs/series/` by default — pass `--pc-dir` for a dataset tag)*

---

### `alignment.py` — dream ↔ form faithfulness
Embeds each dream's text and its generated form in the same space, projects **both** onto
posture axes / design families / symbols, and measures how well the form preserves the
dream's profile — *per-dimension* (which properties survive translation) and *per-dream*
(best-preserved dreams, for the one-dream → one-chair route). Standardizes within modality
and compares to a shuffled baseline.
- **Out:** `outputs/dataset_<tag>/alignment_report.json`, `docs/figures/fig9_alignment.*`
- **Example:** `python scripts/alignment.py --tag pointe`

## The dream → chair bridge

### `posture_map.py` — place forms on the body axes
Projects each form onto the 6 bipolar posture axes and derives chair parameters; prints the
extreme exemplar per axis as a sanity check.
- **Out:** `outputs/dataset_<tag>/posture_profiles.json` (scores + chair params),
  `posture_matrix.json`
- **Example:** `python scripts/posture_map.py --tag pointe`

### `extract_posture.py` — dream text → posture scores
Reads dreams into the 6 axis scores. `clip` = zero-shot projection (no API, short reports);
`llm` = Claude reads the full report against the rubric (best for real reports; needs
`ANTHROPIC_API_KEY`).
- **Key args:** `--method clip|llm`, `--model claude-opus-4-8`, `--limit N`
- **Out:** `outputs/dream_posture_targets.json`
- **Example:** `python scripts/extract_posture.py --method clip`

### `posture_chair.py` — dream → chair (generate & select)
Builds a chair prompt from a dream's target posture, generates K candidates, scores each
one's posture with OpenShape, and keeps the closest match.
- **Key args:** `--ids D030,D076`, `--variants 4`, `--backend point_e|shap_e`, `--steps`
- **Out:** `outputs/chairs/` → `<id>.png` (ranked candidates), `<id>_best.npz/.obj`,
  `chairs_report.json`
- **Example:** `python scripts/posture_chair.py --ids D030,D076 --variants 4`

---

## Inspection & docs

### `render_dreams.py` — eyeball generated forms
Labeled multi-view montages of any set of forms.
- **Key args:** `--tag`, `--ids D003,D027`, `--first N`, `--sample N`, `--exemplars`,
  `--views 4`
- **Out:** `outputs/dataset_<tag>/render_montage.png`
- **Example:** `python scripts/render_dreams.py --tag pointe --first 24`

### `dream_series_demo.py` — original end-to-end demo (8 forms)
Symbol-guided Shap-E generation + classification + clustering on the small `series` set.
- **Out:** `outputs/series/` (results.json, clusters.json, previews, overview.png)
- **Example:** `python scripts/dream_series_demo.py --limit 8`

### `smoke_test.py` — wiring check
Runs embed→classify→cluster on random point clouds (no generator downloads; CLIP weights
download once).
- **Example:** `python scripts/smoke_test.py`

### `md_to_pdf.py` — Markdown → PDF
Renders a Markdown doc to PDF with a Unicode font (used for the handout).
- **Example:** `python scripts/md_to_pdf.py docs/SUMMARY_FOR_STUDENTS.md`
