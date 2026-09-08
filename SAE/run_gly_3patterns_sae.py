"""
Compute Evo2 SAE (all 32768 features, layer-26, evo2_7b_262k) for the
FULL-LENGTH mutant gene sequence of the 38 ClinVar Gly-substitution
candidate variants in evo2/Gly substitution/gly_variant_candidates_3patterns.csv
(edge_fire / edge_dropout / continuous_dropout patterns, COL4A3/4/5).

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_gly_3patterns_sae.py
"""
import os
import run_candidate_variants_sae as base

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gly_3patterns_candidates.csv")

if __name__ == "__main__":
    base.main(csv_path=CSV_PATH, manifest_name="gly_3patterns_manifest.csv")
