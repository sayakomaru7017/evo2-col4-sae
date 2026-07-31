"""
Zero-shot Evo2 delta-likelihood scoring for COL4A5 variants, using the same
method as notebooks/brca1/brca1_zero_shot_vep.ipynb:

  1. Build an 8192-bp window centered on each variant's genomic position.
  2. Build the wild-type window and the variant window (substitution /
     deletion / insertion / duplication applied at the variant site).
  3. Score both with Evo2.score_sequences() (reduce_method='mean' -> average
     per-nucleotide log-likelihood, comparable across different lengths).
  4. evo2_delta_score = var_score - ref_score  (more negative = more disruptive)

Input : XLAS/COL4A5_Evo2_coordinate_list.xlsx, sheet "Evo2_with_coordinates"
        (314 variants: 235 SNV + 79 indel [del/ins/dup/delins]; GRCh38, chrX,
        + strand, coordinates cross-checked with VariantValidator per README)
Output: XLAS/COL4A5_Evo2_delta_scores.csv

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python XLAS/run_col4a5_evo2_vep.py
"""
import os, re, sys, pickle, time
import numpy as np
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
XLSX    = os.path.join(HERE, "COL4A5_Evo2_coordinate_list.xlsx")
OUT_CSV = os.path.join(HERE, "COL4A5_Evo2_delta_scores.csv")
SAE_DIR = os.path.join(os.path.dirname(HERE), "SAE")
PAD_PKL = os.path.join(HERE, "col4a5_padded_sequence.pkl")

WINDOW_SIZE = 8192
PAD         = WINDOW_SIZE // 2 + 100   # extra margin beyond gene boundaries
MODEL_NAME  = "evo2_7b"

# ── Load variant table ────────────────────────────────────────────────────
df = pd.read_excel(XLSX, sheet_name="Evo2_with_coordinates")
print(f"Loaded {len(df)} variants with coordinates")

# ── Reference sequence: COL4A5 gene +/- padding, so every 8192bp window has
#    full context even for variants near the gene boundary (matches the
#    whole-chromosome-fasta approach used previously). ─────────────────────
sys.path.insert(0, SAE_DIR)
import run_col4_sae as sae_base

seqs = pickle.load(open(os.path.join(SAE_DIR, "col4_sequences.pkl"), "rb"))
gene_meta = seqs["COL4A5"]
assert gene_meta["strand"] == "+"

if os.path.exists(PAD_PKL):
    print(f"Loading cached padded sequence from {PAD_PKL}")
    ref_master, master_start = pickle.load(open(PAD_PKL, "rb"))
else:
    master_start = gene_meta["start"] - PAD
    master_end = gene_meta["end"] + PAD
    print(f"Fetching padded COL4A5 region chrX:{master_start:,}-{master_end:,} ...")
    ref_master = sae_base.fetch_ensembl("X", master_start, master_end)
    pickle.dump((ref_master, master_start), open(PAD_PKL, "wb"))
    print(f"  Saved to {PAD_PKL}")

master_len = len(ref_master)
print(f"Master COL4A5 sequence (padded): chrX:{master_start:,}-{master_start+master_len-1:,}  ({master_len:,} bp)")


def local_index(genomic_pos: int) -> int:
    """0-based index into ref_master for a 1-based genomic position."""
    return genomic_pos - master_start


def window_slice(center_local: int):
    s = max(0, center_local - WINDOW_SIZE // 2)
    e = min(master_len, center_local + WINDOW_SIZE // 2)
    return s, e


class RefMismatchError(Exception):
    pass


INS_RE = re.compile(r"ins([ACGTacgt]+)")


def parse_insert_seq(cdna: str) -> str:
    """Concatenate all inserted-base runs following 'ins' in the cDNA string."""
    matches = INS_RE.findall(str(cdna))
    return "".join(matches).upper()


def build_windows(row):
    change_type = row["change_type"]
    start, end = int(row["GRCh38_start"]), int(row["GRCh38_end"])
    cdna = str(row["cDNA"])

    # Known bad coordinate: c.546+2_3insT has a bogus 133-kb "end" (data error).
    # Treat as a single-position insertion at `start`.
    if change_type == "ins" and (end - start + 1) > 2000:
        end = start

    start_local, end_local = local_index(start), local_index(end)
    center_local = (start_local + end_local) // 2
    w_s, w_e = window_slice(center_local)
    ref_window = ref_master[w_s:w_e]
    rel_start, rel_end = start_local - w_s, end_local - w_s

    has_ins = "ins" in cdna
    has_del = "del" in cdna

    if change_type == "SNV":
        ref_base, alt_base = row["ref"], row["alt"]
        if ref_window[rel_start] != ref_base:
            raise RefMismatchError(
                f"ref mismatch at {start}: file says {ref_base}, GRCh38 has "
                f"{ref_window[rel_start]}"
            )
        var_window = ref_window[:rel_start] + alt_base + ref_window[rel_start + 1:]
    elif has_del and has_ins:
        # delins
        ins_seq = parse_insert_seq(cdna)
        var_window = ref_window[:rel_start] + ins_seq + ref_window[rel_end + 1:]
    elif has_del:
        var_window = ref_window[:rel_start] + ref_window[rel_end + 1:]
    elif change_type == "dup":
        dup_seq = ref_window[rel_start:rel_end + 1]
        var_window = ref_window[:rel_end + 1] + dup_seq + ref_window[rel_end + 1:]
    elif has_ins:
        ins_seq = parse_insert_seq(cdna)
        var_window = ref_window[:rel_start + 1] + ins_seq + ref_window[rel_start + 1:]
    else:
        raise ValueError(f"Unhandled change_type={change_type} cDNA={cdna}")

    return ref_window, var_window


def main():
    # 1. Build all windows, dedupe reference windows; separate out rows whose
    #    stated ref base doesn't match GRCh38 (coordinate/strand mismatch --
    #    flagged for manual review rather than guessed at).
    df["evo2_delta_score"] = np.nan
    df["flag"] = ""
    ref_windows, var_windows = [], []
    ref_seq_to_index, ref_indexes, scorable_idx = {}, [], []
    for i, row in df.iterrows():
        try:
            ref_w, var_w = build_windows(row)
        except RefMismatchError as e:
            df.at[i, "flag"] = str(e)
            print(f"  SKIP (flagged) row {i}: {row['cDNA']}: {e}")
            continue
        if ref_w not in ref_seq_to_index:
            ref_seq_to_index[ref_w] = len(ref_windows)
            ref_windows.append(ref_w)
        ref_indexes.append(ref_seq_to_index[ref_w])
        var_windows.append(var_w)
        scorable_idx.append(i)
    ref_indexes = np.array(ref_indexes)
    print(f"Built windows: {len(ref_windows)} unique reference windows, "
          f"{len(var_windows)} variant windows ({len(df) - len(scorable_idx)} flagged/skipped)")

    # 2. Load model and score
    print(f"\nLoading {MODEL_NAME} ...")
    t0 = time.time()
    from evo2.models import Evo2
    model = Evo2(MODEL_NAME)
    print(f"  ready in {time.time()-t0:.1f}s")

    print(f"\nScoring {len(ref_windows)} reference windows ...")
    t0 = time.time()
    ref_scores = model.score_sequences(ref_windows)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"\nScoring {len(var_windows)} variant windows ...")
    t0 = time.time()
    var_scores = model.score_sequences(var_windows)
    print(f"  done in {time.time()-t0:.1f}s")

    # 3. Delta scores
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_indexes]
    df.loc[scorable_idx, "evo2_delta_score"] = delta_scores
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Scored: {len(scorable_idx)}   Flagged/unscored: {len(df) - len(scorable_idx)}")
    print(df[["mut_class", "protein", "cDNA", "change_type", "evo2_delta_score", "flag"]].to_string())


if __name__ == "__main__":
    main()
