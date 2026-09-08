"""
Evo2 delta-likelihood scoring (evo2_7b) for the NEW variants introduced in
XLAS_COL4A5_565例_GRCh38再検証_Evo2用.xlsx that were not already scored in
the previous 478-row Cox解析 cohort (xlas_cox_evo2_7b_delta.csv).

Matches variants by (chr, pos, ref, alt) rather than Analysis_ID (IDs were
partly renamed between the two files). Previously-scored variants are
transcribed (reused) as-is per the user's instruction; only genuinely new
variants are (re)computed here.

Same method as run_xlas_cox_evo2_delta.py: 8192bp window from the padded
COL4A5 reference, VCF-anchor REF/ALT substitution, mean log-likelihood,
delta = var - ref.

Output: XLAS/xlas_565_full_evo2_7b_delta.csv (all 565 rows, merged
        transcribed + newly-scored evo2_delta_score)
"""
import os, pickle, sys, time
import numpy as np
import pandas as pd
import torch
import openpyxl

HERE     = os.path.dirname(os.path.abspath(__file__))
XLSX_NEW = os.path.join(HERE, "XLAS_COL4A5_565例_GRCh38再検証_Evo2用.xlsx")
OLD_CSV  = os.path.join(HERE, "xlas_cox_evo2_7b_delta.csv")
OUT_CSV  = os.path.join(HERE, "xlas_565_full_evo2_7b_delta.csv")
PAD_PKL  = os.path.join(HERE, "col4a5_padded_sequence.pkl")
CKPT     = os.path.join(HERE, "xlas_565_new_score_checkpoint.pkl")

WINDOW_SIZE = 8192
MODEL_NAME  = "evo2_7b"
CHUNK       = 100


def load_all_sheets(xlsx_path):
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    dfs = []
    for sn in wb.sheetnames:
        ws = wb[sn]
        headers = [c.value for c in ws[1]]
        data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
        d = pd.DataFrame(data, columns=headers)
        d["__sheet__"] = sn
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def build_window(ref_master, master_start, pos, ref, alt):
    local = pos - master_start
    s = max(0, local - WINDOW_SIZE // 2)
    e = min(len(ref_master), local + WINDOW_SIZE // 2)
    ref_window = ref_master[s:e]
    rel = local - s
    genome_ref = ref_window[rel:rel + len(ref)]
    assert genome_ref == ref, f"ref mismatch at {pos}: file={ref} genome={genome_ref}"
    var_window = ref_window[:rel] + alt + ref_window[rel + len(ref):]
    return ref_window, var_window


def main():
    print("Loading new 565-row workbook ...")
    new = load_all_sheets(XLSX_NEW)
    new["GRCh38_pos_VCF"] = pd.to_numeric(new["GRCh38_pos_VCF"], errors="coerce")
    elig = new[new["Evo2_eligible"] == "Yes"].copy()
    print(f"  {len(new)} total, {len(elig)} Evo2_eligible")

    print("Loading previous scored cohort ...")
    old = pd.read_csv(OLD_CSV)
    old_scored = old[old["evo2_delta_score"].notna()].copy()
    old_scored["GRCh38_pos"] = pd.to_numeric(old_scored["GRCh38_pos"], errors="coerce")
    score_lookup = {}
    for _, r in old_scored.iterrows():
        key = (r["GRCh38_chr"], int(r["GRCh38_pos"]), r["REF"], r["ALT"])
        score_lookup[key] = r["evo2_delta_score"]
    print(f"  {len(score_lookup)} unique previously-scored variants")

    elig["key"] = list(zip(elig["GRCh38_chr"], elig["GRCh38_pos_VCF"].astype("Int64"), elig["REF"], elig["ALT"]))
    elig["transcribed_score"] = elig["key"].map(lambda k: score_lookup.get((k[0], int(k[1]), k[2], k[3])) if pd.notna(k[1]) else None)
    transcribed = elig[elig["transcribed_score"].notna()]
    need = elig[elig["transcribed_score"].isna()].drop_duplicates(subset="key").copy()
    print(f"  {len(transcribed)} rows transcribed from previous scoring")
    print(f"  {len(need)} unique new variants need scoring")

    ref_master, master_start = pickle.load(open(PAD_PKL, "rb"))

    ref_windows, var_windows = [], []
    ref_seq_to_index, ref_indexes, need_rows = {}, [], []
    for _, row in need.iterrows():
        pos = int(row["GRCh38_pos_VCF"])
        ref_w, var_w = build_window(ref_master, master_start, pos, row["REF"], row["ALT"])
        if ref_w not in ref_seq_to_index:
            ref_seq_to_index[ref_w] = len(ref_windows)
            ref_windows.append(ref_w)
        ref_indexes.append(ref_seq_to_index[ref_w])
        var_windows.append(var_w)
        need_rows.append(row["key"])
    ref_indexes = np.array(ref_indexes)
    print(f"  {len(ref_windows)} unique ref windows, {len(var_windows)} var windows to score")

    all_seqs = ref_windows + var_windows
    n_total = len(all_seqs)

    if n_total > 0:
        if os.path.exists(CKPT):
            scores_done, next_idx = pickle.load(open(CKPT, "rb"))
            print(f"Resuming from checkpoint: {next_idx}/{n_total}")
        else:
            scores_done, next_idx = [], 0

        if next_idx < n_total:
            print(f"Loading {MODEL_NAME} ...")
            t0 = time.time()
            from evo2.models import Evo2
            model = Evo2(MODEL_NAME)
            print(f"  ready in {time.time()-t0:.1f}s")

            while next_idx < n_total:
                chunk = all_seqs[next_idx: next_idx + CHUNK]
                t0 = time.time()
                chunk_scores = model.score_sequences(chunk)
                dt = time.time() - t0
                scores_done.extend(chunk_scores)
                next_idx += len(chunk)
                pickle.dump((scores_done, next_idx), open(CKPT, "wb"))
                print(f"  scored {next_idx}/{n_total}  (+{len(chunk)} in {dt:.1f}s)")

            del model
            torch.cuda.empty_cache()

        all_scores = np.array(scores_done)
        ref_scores = all_scores[:len(ref_windows)]
        var_scores = all_scores[len(ref_windows):]
        new_deltas = var_scores - ref_scores[ref_indexes]
        new_score_lookup = dict(zip(need_rows, new_deltas))
    else:
        new_score_lookup = {}

    elig["evo2_delta_score"] = elig.apply(
        lambda r: r["transcribed_score"] if pd.notna(r["transcribed_score"]) else new_score_lookup.get(r["key"]),
        axis=1,
    )
    elig["score_source"] = elig["transcribed_score"].notna().map({True: "transcribed", False: "newly_scored"})

    out = new.merge(elig[["Analysis_ID", "evo2_delta_score", "score_source"]], on="Analysis_ID", how="left")
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Total rows: {len(out)}  scored: {out['evo2_delta_score'].notna().sum()}")
    print(out["score_source"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
