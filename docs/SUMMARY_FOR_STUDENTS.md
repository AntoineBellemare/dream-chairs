# Dream → Chairs — approaches & design paths

**This is a toolkit of interchangeable ideas.** We've prototyped a
set of building blocks and several ways to connect them. At each
stage there are multiple paths; pick and combine the ones that fit your design intent.

---

## One connective idea

**don't map dream → chair directly — map both
onto body posture** and use that as the shared coordinate. A chair proposes *how to hold
a body*; a dream stages *a body in space*. The common variable is posture. It's a great
organizing principle **if** you want the chair to express the dream's bodily feeling.

---

## The building blocks that exist and work

- **Text → 3D form** — Shap-E (meshes) ✓ and Point-E (point clouds) ✓.
- **A shared 3D↔text space** — OpenShape encodes a form into the *same space as words*, so
  forms and text are directly comparable with no rendering ✓.
- **Descriptor vocabularies** — design families + a large visual dictionary ✓.
- **Clustering + characterization** of a whole corpus ✓.
- **Posture axes** (the dream↔body↔chair bridge) ✓.

---

## Menu of design paths

### 1 · What is the unit of design?
- **Individual dream → one chair** — an idiosyncratic, singular object. ✓
- **A collection → families → an archetype chair** per family (decode the cluster centroid;
  let within-family variance become deformation: spread→asymmetry, instability→tilt). ◦
  The question becomes 'how do we define a family?' and 'in what dimensional space?' (shape, text, posture, symbol).

### 2 · How to find "families" of dreams?
Each clusters something different and yields different families. **They can be combined.**
- **Cluster the 3D shapes themselves** — group the generated point clouds / OpenShape
  embeddings to discover *families of forms* emerging from the data, then let those
  emergent shape-types inform the final results. Data-driven and surprising. ✓
- **Cluster the dream texts** — group by what the reports *say* (text embeddings); families
  by narrative/semantic content rather than form. ◦
- **Cluster by posture** — group by the 6 body-axis coordinates (derived from text or
  shape); families by *bodily attitude* (all the "reclined/cocooned" dreams together). ◦
- **Group by Lakota symbol** — assign each dream its top symbol and cluster by symbol — a
  culturally grounded grouping. Needs Knowledge-Holder descriptors to be real. ◦/✗
- **Hybrid** — e.g. symbol first, then shape-family within each symbol; or posture within
  a shape-family. Encouraged.

### 3 · How to *characterize / name* a family
- **Design-descriptor families** — volume, form language, symmetry, edge, proportion,
  surface, posture — via contrastive signatures (what a cluster has *more* of than
  average). ✓
- **Emergent visual descriptors** — a large neutral dictionary, data-driven. ✓
- **LLM labels** — generate a short text label for each cluster (e.g. "tall spire", "cocooned
  pod"). ✓
- **Lakota symbols + their qualities** — name families by symbol and the spatial qualities
  attached to it (cultural grounding). ◦/✗

### 4 · Linking dream → chair (the body)
- **Posture from the dream text** (LLM rubric for nuanced reports, or zero-shot CLIP for
  short ones) → chair parameters. ✓
- **Posture from the 3D shape** (project the form onto the body axes) → chair parameters. ✓
- **Generate-and-select** — generate several chairs from the posture, keep the one whose
  body best matches the dream. ✓
- **Archetype chair** — the chair of all "falling" dreams, from a cluster centroid. ◦

### 5 · Representation — point cloud vs mesh
- **Point cloud** for the dream-logic / ambiguity stage (density itself = liminality). ✓
- **Mesh** for the fabricable, body-supporting object. ✓
- Use both sequentially, or pick one.

---

## Open knobs

These are where design + phenomenology judgment should drive the system.

- **A · Posture axes** (`data/descriptors/body_posture_axes.json`) — are the 6 axes, their
  poles, and the chair-parameter mappings right? Check the axis-extreme sheet; each
  hand-score ~10 dreams to compare against the auto-extraction.
- **B · Which substrate defines a family** — shapes, text, posture, or symbols ?
- **C · Design descriptors that name clusters** (`data/descriptors/design_families.json`,
  `visual_descriptors.json`) — right vocabularies? Missing categories (seat geometry, leg
  structure, back profile)? Edit and re-cluster easily.

---
