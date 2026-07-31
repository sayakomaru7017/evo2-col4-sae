"""
Compute Evo2 SAE (all 32768 features, layer-26) for the FULL-LENGTH mutant
gene sequence of the 19 Alport NC1-domain candidate variants
(SAE/nc1_domain_candidates.csv): 14 pathogenic + 5 benign controls across
COL4A3/COL4A4/COL4A5.

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_nc1_domain_sae.py
"""
import os
import run_candidate_variants_sae as base

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nc1_domain_candidates.csv")

if __name__ == "__main__":
    base.main(csv_path=CSV_PATH, manifest_name="nc1_domain_manifest.csv")
