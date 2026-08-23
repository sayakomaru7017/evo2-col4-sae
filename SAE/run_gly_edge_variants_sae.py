"""
Compute Evo2 SAE (all 32768 features, layer-26, evo2_7b_262k) for the
FULL-LENGTH mutant gene sequence of 15 Gly-edge candidate variants
(SAE/gly_edge_variants_candidates.csv): f11735 edge-fire, edge-dropout, and
continuous-vs-dropout comparison sets across COL4A3/4/5. No delta-likelihood
scoring performed (not needed for this batch).

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_gly_edge_variants_sae.py
"""
import os
import run_candidate_variants_sae as base

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gly_edge_variants_candidates.csv")

if __name__ == "__main__":
    base.main(csv_path=CSV_PATH, manifest_name="gly_edge_variants_manifest.csv")
