"""
Independently verify the 436 Evo2-eligible XLAS COL4A5 variants (Cox解析
sheet) by reverse-mapping their recorded GRCh38 genomic coordinates
(GRCh38_pos/REF/ALT) through VariantValidator against NM_000495.5, and
comparing VariantValidator's predicted transcript/protein consequence
against the recorded Genomic_variant_type / Functional_variant_type /
Truncating_status / c.HGVS_current / p.HGVS_current columns.

This catches errors that a simple "does REF match the reference genome"
check cannot: e.g. a coordinate that is internally self-consistent (REF
matches GRCh38) but does not correspond to the variant the source paper
actually described (old-numbering transcript offset issues, wrong exon,
etc).

Output: XLAS/xlas_coordinate_verification.csv
"""
import os, re, time, json, pickle
import requests
import pandas as pd
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "組入症例＿XLAS_strict_analysis_population_v2_原著再監査_GRCh38_最終.xlsx")
OUT_CSV = os.path.join(HERE, "xlas_coordinate_verification.csv")
CACHE_PKL = os.path.join(HERE, "variantvalidator_cache.pkl")

TRANSCRIPT = "NM_000495.5"
VV_BASE = "https://rest.variantvalidator.org/VariantValidator/variantvalidator/hg38"


def load_sheet(sheet_name):
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]
    data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
    return pd.DataFrame(data, columns=headers)


def vcf_to_genomic_hgvs(pos, ref, alt):
    """Convert VCF-anchor-style (pos,ref,alt) to a g. HGVS description on chrX (GRCh38)."""
    # trim common prefix
    i = 0
    while i < len(ref) and i < len(alt) and ref[i] == alt[i]:
        i += 1
    ref2, alt2 = ref[i:], alt[i:]
    pos2 = pos + i
    # trim common suffix
    j = 0
    while j < len(ref2) and j < len(alt2) and ref2[-1-j] == alt2[-1-j] and (len(ref2)-j) > 0 and (len(alt2)-j) > 0:
        j += 1
    if j > 0:
        ref2, alt2 = ref2[:len(ref2)-j], alt2[:len(alt2)-j]

    if len(ref2) == 0 and len(alt2) == 0:
        return None  # no variant (shouldn't happen)
    if len(ref2) == 1 and len(alt2) == 1:
        return f"NC_000023.11:g.{pos2}{ref2}>{alt2}"
    if len(alt2) == 0:
        # pure deletion of ref2, starting at pos2
        if len(ref2) == 1:
            return f"NC_000023.11:g.{pos2}del"
        return f"NC_000023.11:g.{pos2}_{pos2+len(ref2)-1}del"
    if len(ref2) == 0:
        # pure insertion between pos2-1 and pos2
        return f"NC_000023.11:g.{pos2-1}_{pos2}ins{alt2}"
    # delins
    if len(ref2) == 1:
        return f"NC_000023.11:g.{pos2}delins{alt2}"
    return f"NC_000023.11:g.{pos2}_{pos2+len(ref2)-1}delins{alt2}"


def query_vv(g_hgvs, cache):
    if g_hgvs in cache:
        return cache[g_hgvs]
    url = f"{VV_BASE}/{g_hgvs}/{TRANSCRIPT}?content-type=application/json"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            d = r.json()
            cache[g_hgvs] = d
            return d
        except Exception as e:
            print(f"    retry {attempt+1} for {g_hgvs}: {e}")
            time.sleep(2)
    cache[g_hgvs] = {"error": "failed after retries"}
    return cache[g_hgvs]


def classify_consequence(tlr):
    """Rough classification from VariantValidator's protein tlr string."""
    if tlr is None:
        return "unknown"
    if "Ter" in tlr or "*" in tlr:
        return "truncating"
    if "fs" in tlr:
        return "frameshift"
    if "=" in tlr:
        return "synonymous"
    if re.search(r"p\.\(?[A-Za-z]{3}\d+[A-Za-z]{3}\)?$", tlr):
        return "missense"
    return "other"


def main():
    print("Loading Cox解析 sheet ...")
    df = load_sheet("Cox解析")
    df["GRCh38_pos_int"] = pd.to_numeric(df["GRCh38_pos"], errors="coerce")
    elig = df[df["Evo2_eligible"] == "Yes"].copy()
    print(f"  {len(elig)} eligible rows to verify")

    cache = pickle.load(open(CACHE_PKL, "rb")) if os.path.exists(CACHE_PKL) else {}

    results = []
    for n, (i, row) in enumerate(elig.iterrows()):
        pos, ref, alt = int(row["GRCh38_pos_int"]), row["REF"], row["ALT"]
        g_hgvs = vcf_to_genomic_hgvs(pos, ref, alt)
        d = query_vv(g_hgvs, cache)

        rec = dict(
            Analysis_ID=row["Analysis_ID"], Source=row["Source"],
            c_HGVS_current=row["c.HGVS_current"], p_HGVS_current=row["p.HGVS_current"],
            GRCh38_pos=pos, REF=ref, ALT=alt, g_hgvs=g_hgvs,
            Genomic_variant_type=row["Genomic_variant_type"],
            Truncating_status=row["Truncating_status"],
        )

        main_key = next((k for k in d if k not in ("flag", "metadata") and isinstance(d.get(k), dict)
                          and "primary_assembly_loci" in d[k]), None)
        if main_key is None:
            rec["VV_status"] = "FAIL"
            rec["VV_warning"] = str(d.get("validation_warning_1", d))[:300]
            rec["VV_c_hgvs"] = None
            rec["VV_protein"] = None
            rec["VV_consequence_class"] = None
        else:
            entry = d[main_key]
            rec["VV_status"] = "OK"
            rec["VV_warning"] = ""
            rec["VV_c_hgvs"] = entry.get("hgvs_transcript_variant")
            tlr = entry.get("hgvs_predicted_protein_consequence", {}).get("tlr")
            rec["VV_protein"] = tlr
            rec["VV_consequence_class"] = classify_consequence(tlr)
            vcf = entry.get("primary_assembly_loci", {}).get("hg38", {}).get("vcf", {})
            rec["VV_vcf_pos"] = vcf.get("pos")
            rec["VV_vcf_ref"] = vcf.get("ref")
            rec["VV_vcf_alt"] = vcf.get("alt")

        results.append(rec)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(elig)} done")
            pickle.dump(cache, open(CACHE_PKL, "wb"))

    pickle.dump(cache, open(CACHE_PKL, "wb"))
    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print("\nVV_status counts:")
    print(out["VV_status"].value_counts())


if __name__ == "__main__":
    main()
