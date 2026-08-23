"""
Compute Evo2 SAE (all 32768 features, layer-26, evo2_7b_262k) for the
FULL-LENGTH mutant gene sequence of 4 extra candidate variants
(SAE/extra_variants_candidates.csv): COL4A5 G953V, COL4A3 G43R,
COL4A4 G545A, COL4A4 G999E.

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_extra_variants_sae.py
"""
import os
import run_candidate_variants_sae as base

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_variants_candidates.csv")

if __name__ == "__main__":
    base.main(csv_path=CSV_PATH, manifest_name="extra_variants_manifest.csv")
