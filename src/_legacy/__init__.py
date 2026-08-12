"""Legacy / archived source modules.

These files are retired from the formal v0.4 IAH path:
  - hch_v2.py / hch_v2_data.py : legacy HCH v2 (BiOMC, ContinuousStateHead,
      CARA/KL calibration). Superseded by hch_v2_pipeline.py + iah_* + w1_*.
  - bech.py / selective_hurdle.py : v1 HCH / BECH. Superseded by v2.
  - audit_peer_gain.py / make_evidence.py / run_peer_baselines.py : v1 tooling.

They are kept only for historical-experiment reproducibility. The formal
runner (hch_v2_pipeline.py) must NEVER import anything from here.
"""
