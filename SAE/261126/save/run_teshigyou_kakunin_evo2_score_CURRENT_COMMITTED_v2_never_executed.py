"""
Compute Evo2 delta-likelihood scores FROM SCRATCH (evo2_7b, no transcription/reuse
from any previous scoring run) for all 400 rows in
臨床重症度/手作業確認済み.xlsx (sheet "S作成all症例").

IMPORTANT: different VCF-anchor representations of the *same* underlying edit
(e.g. c.1474_1475delinsTAT written as chrX/108595559/CC/TAT vs
chrX/108595558/GCC/GTAT) produce byte-identical mutant sequences but, if the
8192bp scoring window is centered on the raw recorded position, the window
boundaries shift by the anchor offset and Evo2 returns slightly different
scores for what is actually one variant. To avoid this, every (pos,ref,alt)
is first normalized (left-aligned + trimmed, standard vt-normalize/bcftools
norm algorithm) against the reference before being used as the dedup key and
before building the scoring window, so logically identical edits always get
exactly one score.

Method: 8192bp window centered on the NORMALIZED position, VCF-anchor
REF/ALT substitution, mean log-likelihood via model.score_sequences(),
delta = var_score - ref_score.

Output: 臨床重症度/手作業確認済み_Evo2score付与.xlsx (same sheet/columns,
Evo2score column filled in).
"""
import os, pickle, time
import numpy as np
import pandas as pd
import torch
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX_IN = os.path.join(HERE, "臨床重症度", "手作業確認済み.xlsx")
XLSX_OUT = os.path.join(HERE, "臨床重症度", "手作業確認済み_Evo2score付与.xlsx")
PAD_PKL = os.path.join(HERE, "col4a5_padded_sequence.pkl")
CKPT = os.path.join(HERE, "teshigyou_kakunin_score_checkpoint_v2.pkl")

WINDOW_SIZE = 8192
MODEL_NAME = "evo2_7b"
CHUNK = 100
SHEET = "S作成all症例"


def get_base(ref_master, master_start, pos):
    return ref_master[pos - master_start]


def normalize(ref_master, master_start, pos, ref, alt):
    """Left-align + trim (vt-normalize / bcftools norm style) so that
    equivalent VCF-anchor representations of the same edit collapse to one
    canonical (pos, ref, alt)."""
    ref, alt = list(ref), list(alt)
    while True:
        if ref and alt and ref[-1] == alt[-1] and (len(ref) > 1 or len(alt) > 1):
            ref.pop(); alt.pop()
            if not ref or not alt:
                pos -= 1
                pb = get_base(ref_master, master_start, pos)
                ref.insert(0, pb); alt.insert(0, pb)
            continue
        break
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref.pop(0); alt.pop(0); pos += 1
    return pos, "".join(ref), "".join(alt)


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
    wb = openpyxl.load_workbook(XLSX_IN, data_only=True)
    ws = wb[SHEET]
    headers = [c.value for c in ws[1]]
    data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
    df = pd.DataFrame(data, columns=headers)
    print(f"Loaded {len(df)} rows")

    ref_master, master_start = pickle.load(open(PAD_PKL, "rb"))

    df["norm_key"] = df.apply(
        lambda r: (r["GRCh38_chr"],) + normalize(ref_master, master_start, int(r["GRCh38_pos_VCF"]), r["REF"], r["ALT"]),
        axis=1,
    )
    unique_keys = df.drop_duplicates(subset="norm_key")["norm_key"].tolist()
    print(f"{len(unique_keys)} unique NORMALIZED variants to score (from {len(df)} rows, "
          f"vs {df.drop_duplicates(subset=['GRCh38_chr','GRCh38_pos_VCF','REF','ALT']).shape[0]} unique raw tuples)")

    ref_windows, var_windows = [], []
    ref_seq_to_index, ref_indexes = {}, []
    for (chrom, pos, ref, alt) in unique_keys:
        ref_w, var_w = build_window(ref_master, master_start, pos, ref, alt)
        if ref_w not in ref_seq_to_index:
            ref_seq_to_index[ref_w] = len(ref_windows)
            ref_windows.append(ref_w)
        ref_indexes.append(ref_seq_to_index[ref_w])
        var_windows.append(var_w)
    ref_indexes = np.array(ref_indexes)
    print(f"{len(ref_windows)} unique ref windows, {len(var_windows)} var windows")

    all_seqs = ref_windows + var_windows
    n_total = len(all_seqs)

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
    deltas = var_scores - ref_scores[ref_indexes]
    key_to_score = dict(zip(unique_keys, deltas))

    df["Evo2score"] = df["norm_key"].map(key_to_score)
    df = df.drop(columns=["norm_key"])
    print(f"\nScored rows: {df['Evo2score'].notna().sum()} / {len(df)}")

    wb_out = openpyxl.Workbook()
    wb_out.remove(wb_out.active)
    ws_out = wb_out.create_sheet(SHEET)
    ws_out.append(headers)
    for _, row in df.iterrows():
        ws_out.append([row[h] if h in df.columns else None for h in headers])
    wb_out.save(XLSX_OUT)
    print(f"Saved: {XLSX_OUT}")


if __name__ == "__main__":
    main()
