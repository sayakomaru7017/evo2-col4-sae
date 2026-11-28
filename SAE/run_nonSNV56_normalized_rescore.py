#!/usr/bin/env python3
"""Evo2 non-SNV rescore for the 56 left-aligned/parsimony-normalized variants
(sheet "非SNV_再計算用56変異" of XLAS_400症例_左寄せ正規化_変更箇所色分け_20261128.xlsx).

Identical method/environment checks to the 2026-11-28 run01
(02_XLAS_nonSNV_rescore.py): same evo2_7b checkpoint/revision check, same
window-construction convention (center on the variant, add/remove flanking
context symmetrically so WT and mutant are both exactly 8,192bp, odd
remainder assigned to the right), same scoring call
(model.score_sequences([seq]), batch_size=1, prepend_bos=False,
reduce_method='mean', average_reverse_complement=False), same reference
sequence (col4a5_padded_sequence.pkl, sha256 verified against run01).

POS/REF/ALT are taken AS-IS from the Excel sheet -- no re-normalization or
coordinate changes are performed here.
"""
import argparse, csv, hashlib, json, math, re, time, sys, os, subprocess
from pathlib import Path

W = 8192
START = 108435549
END = 108701743
MODEL = 'evo2_7b'
EXPECTED_REV = 'bda0089f92582d5baabf0f22d9fc85f3588f6b58'
EXPECTED_REF_SHA256 = '908c7f134174e6e6db6aa75243ee01c3ed098ce84f7c8d6bc46985ea83737ce9'

XLSX = "/home/azureuser/Evo2/delta likelihoods/non-SNV_57variants/左寄せ/XLAS_400症例_左寄せ正規化_変更箇所色分け_20261128.xlsx"
REF_PKL = "/home/azureuser/Evo2/XLAS/col4a5_padded_sequence.pkl"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(s):
    return hashlib.sha256(s.encode('ascii')).hexdigest()


def file_sha(p):
    with open(p, 'rb') as f:
        h = hashlib.sha256()
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_csv(p, rows):
    require(bool(rows), 'Empty rows')
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(p)


def split_change(d):
    left = (abs(d) // 2) * (1 if d >= 0 else -1)
    return left, d - left


def build(seq, start, pos, ref, alt):
    """Identical logic to 02_XLAS_nonSNV_rescore.py's build()."""
    i = pos - start
    d = len(alt) - len(ref)
    l, r = split_change(d)
    s = i - W // 2
    e = i + W // 2
    require(0 <= i and i + len(ref) <= len(seq), 'REF outside reference')
    require(seq[i:i + len(ref)] == ref, 'REF mismatch')
    ms = s + l
    me = e - r
    require(0 <= min(s, ms) and max(e, me) <= len(seq), 'Insufficient flanking reference')
    require(ms <= i and i + len(ref) <= me, 'Allele outside adjusted window')
    wt = seq[s:e]
    mt = seq[ms:i] + alt + seq[i + len(ref):me]
    require(len(wt) == len(mt) == W, 'Incorrect window length')
    require(re.fullmatch('[ACGT]+', wt) and re.fullmatch('[ACGT]+', mt), 'Non-ACGT in input window')
    require(mt[i - ms:i - ms + len(alt)] == alt, 'ALT not at intended position')
    a = min(s, ms)
    b = max(e, me)
    extended = seq[a:i] + alt + seq[i + len(ref):b]
    require(extended[ms - a:ms - a + W] == mt, 'Independent mutation/crop check failed')
    return wt, mt, dict(ref_start=s + start, ref_end=e + start - 1,
                         mut_source_start=ms + start, mut_source_end=me + start - 1,
                         net_change=d, left_trim=l, right_trim=r,
                         ref_allele_index0=i - s, mut_allele_index0=i - ms)


def load_variants():
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb["非SNV_再計算用56変異"]
    headers = [c.value for c in ws[1]]
    data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
    rows = [dict(zip(headers, r)) for r in data]
    require(len(rows) == 56, f'Expected 56 variants, got {len(rows)}')
    require(len({r['Normalized_ID'] for r in rows}) == 56, 'Duplicate Normalized_ID')
    keys = {(r['GRCh38_chr'], int(r['GRCh38_pos_VCF']), r['REF'], r['ALT']) for r in rows}
    require(len(keys) == 56, 'Duplicate variant key')
    require(sum(int(r['n_cases']) for r in rows) == 65, f"Expected 65 cases, got {sum(int(r['n_cases']) for r in rows)}")
    return rows


def prepare(out_dir):
    import pickle
    seq, start = pickle.load(open(REF_PKL, 'rb'))
    require(start == START and len(seq) == END - START + 1, 'Unexpected reference interval')
    require(sha(seq) == EXPECTED_REF_SHA256, 'Reference sequence sha256 mismatch vs run01')

    rows = load_variants()
    out_dir.mkdir(parents=True, exist_ok=False)

    qc = []
    records = []
    for row in rows:
        vid = row['Normalized_ID']
        pos = int(row['GRCh38_pos_VCF'])
        ref = row['REF']
        alt = row['ALT']
        require(row['GRCh38_chr'] == 'chrX' and ref != alt, f'{vid}: invalid variant')
        require(re.fullmatch('[ACGT]+', ref) and re.fullmatch('[ACGT]+', alt), f'{vid}: invalid allele chars')
        i = pos - start
        require(0 <= i < len(seq) and seq[i:i + len(ref)] == ref, f'{vid}: REF mismatch vs reference genome')
        wt, mt, meta = build(seq, start, pos, ref, alt)
        require(meta['net_change'] == int(row['net_length_change']),
                f"{vid}: net_length_change mismatch (sheet={row['net_length_change']}, computed={meta['net_change']})")
        records.append(dict(row=row, wt=wt, mt=mt, meta=meta))
        qc.append(dict(variant_id=vid, pos=pos, ref=ref, alt=alt,
                        ref_match=True, wt_len=len(wt), mt_len=len(mt),
                        net_change=meta['net_change'], status='PASS',
                        wt_sha256=sha(wt), mt_sha256=sha(mt)))

    write_csv(out_dir / 'QC.csv', qc)
    write_csv(out_dir / 'windows.csv', [
        dict(variant_id=x['row']['Normalized_ID'], **x['meta'], ref_len=W, mut_len=W,
             ref_sha256=sha(x['wt']), mut_sha256=sha(x['mt']))
        for x in records
    ])
    with (out_dir / 'input_sequences.fasta').open('w') as f:
        for x in records:
            vid = x['row']['Normalized_ID']
            f.write(f">{vid}_wt\n{x['wt']}\n>{vid}_mt\n{x['mt']}\n")

    manifest = dict(reference_start=start, reference_end=start + len(seq) - 1,
                     reference_sha256=sha(seq), records=records)
    (out_dir / 'prepared.json').write_text(json.dumps(manifest, indent=2))
    print(f'QC: REF 56/56 PASS; all windows WT=MT={W}bp -> {out_dir}', flush=True)
    return records


def cache_check(hub):
    hub = Path(hub)
    repo = hub / 'models--arcinstitute--evo2_7b'
    main = repo / 'refs/main'
    require(main.is_file(), 'Missing cached refs/main')
    rev = main.read_text().strip()
    require(rev == EXPECTED_REV, f'Cached main revision mismatch: {rev}')
    snap = repo / 'snapshots' / rev
    require(snap.is_dir(), 'Missing expected snapshot')
    require((snap / 'config.json').is_file(), 'Missing cached config.json')
    merged = hub.parent / (MODEL + '.pt')
    full = snap / (MODEL + '.pt')
    if merged.is_file():
        weights = merged
        provenance = 'existing merged weight; revision inferred from cache'
        if full.is_file():
            require(merged.stat().st_size == full.stat().st_size and
                    (os.path.samefile(merged, full) or file_sha(merged) == file_sha(full)),
                    'Merged weight differs from expected snapshot')
            provenance = 'merged weight SHA256 matches expected snapshot complete weight'
    elif full.is_file():
        weights = full
        provenance = 'complete weight in expected snapshot'
    else:
        raise ValueError('No complete cached weight; do not download or merge on paid VM')
    require(weights.stat().st_size > 10_000_000_000, 'Cached weight unexpectedly small')
    return dict(cache_root=str(hub), cache_revision=rev, snapshot=str(snap),
                weights=str(weights), weight_bytes=weights.stat().st_size,
                weight_provenance=provenance)


def score(out_dir, minutes):
    t = time.monotonic()
    manifest = json.loads((out_dir / 'prepared.json').read_text())
    records = manifest['records']
    require(not any((out_dir / f).exists() for f in
                     ('runtime.json', 'scores_PARTIAL.csv', 'scores_COMPLETE.csv')),
            'Existing run files present: use a new output directory')
    require(os.environ.get('HF_HUB_OFFLINE') == '1', 'Set HF_HUB_OFFLINE=1 before launching Python')
    from huggingface_hub import constants
    cached = cache_check(constants.HF_HUB_CACHE)
    print(cached, flush=True)
    require(minutes > 0, 'Positive time budget required')

    import importlib.metadata as md, inspect
    import torch
    from evo2.models import Evo2
    versions = {k: md.version(k) for k in ('evo2', 'vtx')}
    require(versions == {'evo2': '0.6.0', 'vtx': '1.1.0'}, f'Unexpected package versions: {versions}')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    sig = inspect.signature(Evo2.score_sequences)
    for k, v in dict(batch_size=1, prepend_bos=False, reduce_method='mean',
                      average_reverse_complement=False).items():
        require(k in sig.parameters and sig.parameters[k].default == v, f'Unexpected scoring default: {k}')
    ctor = inspect.signature(Evo2)
    require('use_kernels' in ctor.parameters and ctor.parameters['use_kernels'].default is False,
            'Unexpected kernel setting')
    import evo2.models as em
    require(getattr(em, 'HAS_TE', None) is False, 'Expected historical bf16 fallback (HAS_TE=False)')
    from evo2.utils import CONFIG_MAP
    require(CONFIG_MAP[MODEL].endswith('evo2-7b-1m.yml'), 'Unexpected 7B config path')

    info = dict(model=MODEL, **cached, versions=versions, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), signature=str(sig),
                HF_HUB_OFFLINE=os.environ['HF_HUB_OFFLINE'],
                script_sha256=file_sha(Path(__file__)),
                prepared_sha256=file_sha(out_dir / 'prepared.json'),
                loader_sha256=file_sha(inspect.getfile(Evo2)),
                ctor_signature=str(ctor), HAS_TE=em.HAS_TE)
    for name, cmd in [('pip_freeze.txt', [sys.executable, '-m', 'pip', 'freeze']),
                       ('nvidia_smi.txt', ['nvidia-smi'])]:
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            (out_dir / name).write_text(cp.stdout + cp.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            (out_dir / name).write_text(str(exc))
    (out_dir / 'runtime.json').write_text(json.dumps(info, indent=2))
    print(info, flush=True)

    load_t = time.monotonic()
    model = Evo2(MODEL)
    print(f"Model load: {time.monotonic()-load_t:.1f}s; elapsed: {time.monotonic()-t:.1f}s", flush=True)

    def one(seq):
        require(time.monotonic() - t < minutes * 60, 'Time budget reached; preserve saved outputs and notify user')
        one_t = time.monotonic()
        v = float(model.score_sequences([seq])[0])
        print(f"Sequence {len(seq)}bp: {time.monotonic()-one_t:.2f}s; elapsed: {time.monotonic()-t:.1f}s", flush=True)
        require(math.isfinite(v), 'Nonfinite model score')
        return v

    results = []
    for x in records:
        row = x['row']
        vid = row['Normalized_ID']
        ref_ll = one(x['wt'])
        var_ll = one(x['mt'])
        delta = var_ll - ref_ll
        result = dict(
            Normalized_ID=vid, GRCh38_chr=row['GRCh38_chr'], GRCh38_pos_VCF=row['GRCh38_pos_VCF'],
            REF=row['REF'], ALT=row['ALT'], n_cases=row['n_cases'],
            Previous_variant_IDs=row['Previous_variant_IDs'],
            net_length_change=x['meta']['net_change'],
            ref_ll=ref_ll, var_ll=var_ll, Evo2score=delta,
            wt_len=len(x['wt']), mt_len=len(x['mt']),
            ref_start=x['meta']['ref_start'], ref_end=x['meta']['ref_end'],
            mut_source_start=x['meta']['mut_source_start'], mut_source_end=x['meta']['mut_source_end'],
            left_trim=x['meta']['left_trim'], right_trim=x['meta']['right_trim'],
            wt_sha256=sha(x['wt']), mt_sha256=sha(x['mt']),
        )
        results.append(result)
        write_csv(out_dir / 'scores_PARTIAL.csv', results)
        print(f"{len(results)}/56 {vid} delta={delta:.9g}", flush=True)

    write_csv(out_dir / 'scores_COMPLETE.csv', results)
    print('COMPLETE 56/56; save outputs and notify user. User will manually deallocate VM; AI must not stop VM.', flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--out', required=True)
    b = sub.add_parser('score')
    b.add_argument('--out', required=True)
    b.add_argument('--minutes', type=float, default=35)
    args = p.parse_args()
    if args.cmd == 'prepare':
        prepare(Path(args.out))
    elif args.cmd == 'score':
        score(Path(args.out), args.minutes)


if __name__ == '__main__':
    main()
