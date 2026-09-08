"""
Independently re-derive GRCh38 coordinates for all 403 rows in the
"all症例" sheet of "All 論文除外済み.xlsx", using ONLY the trustworthy
paper-derived columns (A-M: Original_variant_notation,
Original_protein_notation, c.HGVS_NM_000495.5) -- NOT the existing
GRCh38_pos_VCF/REF/ALT columns (N onward), which are not verified.

For each unique c.HGVS_NM_000495.5, forward-query VariantValidator
(NM_000495.5:c.XXX -> GRCh38 g.) to get the "ground truth" GRCh38
position/ref/alt, then compare against what's currently recorded in
the sheet to find discrepancies.

Output: XLAS/all_soorei_coordinate_reverification.csv
"""
import os, re, time, pickle
import requests, urllib.parse
import pandas as pd
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = "/home/azureuser/Evo2/XLAS/臨床重症度/All 論文除外済み.xlsx"
OUT_CSV = os.path.join(HERE, "all_soorei_coordinate_reverification.csv")
CACHE_PKL = os.path.join(HERE, "variantvalidator_forward_cache.pkl")

TRANSCRIPT = "NM_000495.5"
VV_BASE = "https://rest.variantvalidator.org/VariantValidator/variantvalidator/hg38"


def normalize_hgvs(s: str) -> str:
    s = s.strip()
    # uppercase nucleotide letters that follow a digit or '>' (crude but effective
    # for this dataset's typical forms: c.123g>a, c.123delg, etc.)
    return re.sub(r"([ACGTacgt]+)$", lambda m: m.group(1).upper(), s)


def query_vv_forward(chgvs: str, cache: dict, max_retries=6):
    key = chgvs
    if key in cache:
        return cache[key]
    variants_to_try = [chgvs]
    # common redundant-suffix fix: c.123delG -> c.123del
    m = re.match(r"^(c\.\d+(?:_\d+)?del)[ACGT]+$", chgvs)
    if m:
        variants_to_try.append(m.group(1))
    m2 = re.match(r"^(c\.\d+(?:_\d+)?del)[ACGT]+(ins[ACGT]+)$", chgvs)
    if m2:
        variants_to_try.append(m2.group(1) + m2.group(2))

    last_d = None
    for variant in variants_to_try:
        url = f"{VV_BASE}/{urllib.parse.quote(f'{TRANSCRIPT}:{variant}')}/{TRANSCRIPT}?content-type=application/json"
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=30)
                d = r.json()
            except Exception as e:
                d = {"error": str(e)}
            if isinstance(d, dict) and "Rate limit" in str(d.get("message", "")):
                wait = 5 * (attempt + 1)
                time.sleep(wait)
                continue
            last_d = d
            break
        main_key = next((k for k in d if k not in ("flag", "metadata")
                          and isinstance(d.get(k), dict) and "primary_assembly_loci" in d[k]), None) if isinstance(d, dict) else None
        if main_key:
            cache[key] = d
            return d
        time.sleep(0.3)
    cache[key] = last_d
    return last_d


def extract_vcf(d):
    if not isinstance(d, dict):
        return None
    main_key = next((k for k in d if k not in ("flag", "metadata")
                      and isinstance(d.get(k), dict) and "primary_assembly_loci" in d[k]), None)
    if main_key is None:
        return None, None
    loci = d[main_key]["primary_assembly_loci"]
    vcf = loci.get("hg38", {}).get("vcf")
    if not vcf:
        return None, None
    return (int(vcf["pos"]), vcf["ref"], vcf["alt"]), d[main_key].get("hgvs_transcript_variant")


def main():
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["all症例"]
    headers = [c.value for c in ws[1]]
    data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
    df = pd.DataFrame(data, columns=headers)
    df["GRCh38_pos_VCF"] = pd.to_numeric(df["GRCh38_pos_VCF"], errors="coerce")

    cache = pickle.load(open(CACHE_PKL, "rb")) if os.path.exists(CACHE_PKL) else {}

    unique_hgvs = df["c.HGVS_NM_000495.5"].dropna().unique()
    print(f"{len(unique_hgvs)} unique c.HGVS_NM_000495.5 values to verify")

    results = {}
    for i, chgvs in enumerate(unique_hgvs):
        norm = normalize_hgvs(str(chgvs))
        d = query_vv_forward(norm, cache)
        vcf_result = extract_vcf(d)
        results[chgvs] = vcf_result
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(unique_hgvs)} done")
            pickle.dump(cache, open(CACHE_PKL, "wb"))
        time.sleep(0.4)
    pickle.dump(cache, open(CACHE_PKL, "wb"))

    rows = []
    for _, row in df.iterrows():
        chgvs = row["c.HGVS_NM_000495.5"]
        vv_result = results.get(chgvs)
        if vv_result is None or vv_result[0] is None:
            vv_pos, vv_ref, vv_alt, vv_chgvs = None, None, None, None
            status = "VV_FAIL"
        else:
            (vv_pos, vv_ref, vv_alt), vv_chgvs = vv_result
            status = "OK"

        rec_pos = row["GRCh38_pos_VCF"]
        rec_ref, rec_alt = row["REF"], row["ALT"]
        match = (status == "OK" and vv_pos == rec_pos and vv_ref == rec_ref and vv_alt == rec_alt)

        rows.append(dict(
            Analysis_ID=row["Analysis_ID"], Source=row["Source"],
            Original_variant_notation=row["Original_variant_notation"],
            Original_protein_notation=row["Original_protein_notation"],
            c_HGVS_NM_000495_5=chgvs,
            recorded_pos=rec_pos, recorded_ref=rec_ref, recorded_alt=rec_alt,
            vv_status=status, vv_chgvs=vv_chgvs,
            vv_pos=vv_pos, vv_ref=vv_ref, vv_alt=vv_alt,
            match=match,
        ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(out["vv_status"].value_counts())
    print("match value counts:")
    print(out["match"].value_counts())


if __name__ == "__main__":
    main()
