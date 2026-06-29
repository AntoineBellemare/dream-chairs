"""dream-chairs: dream text -> 3D shape (Point-E / Shap-E) -> classify against symbol descriptors.

Pipeline stages live in sibling modules:
  generate.py  text -> point cloud / mesh
  embed.py     shape <-> text into a shared embedding space
  classify.py  cosine similarity vs. symbol descriptors
  cluster.py   emergent visual clusters across dreams
  pipeline.py  orchestration
"""

__version__ = "0.1.0"
