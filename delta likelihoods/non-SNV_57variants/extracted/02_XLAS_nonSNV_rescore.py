#!/usr/bin/env python3
"""Evo2 non-SNV rescore. A.3.12, Nature supplementary methods (2026).
Implementation choice: keep original VCF-anchor-centered reference window;
mutant flanks adjusted symmetrically, odd extra base assigned to right.
This is our documented convention, not a verified author implementation.
CPU prepare and GPU score are separate; no network/package installation.
"""
import argparse, csv, hashlib, json, math, re, time, sys, os, subprocess
from pathlib import Path
W=8192
START=108435549
END=108701743
MODEL='evo2_7b'
EXPECTED_REV='bda0089f92582d5baabf0f22d9fc85f3588f6b58'
KEY=['GRCh38_chr','GRCh38_pos_VCF','REF','ALT']
CONTROL_IDS=('NSV017','NSV026','NSV015')
CONTROL_CHANGES=(-21,-1,18)

def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(s): return hashlib.sha256(s.encode('ascii')).hexdigest()
def read_csv(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    require(bool(rows),'Empty rows'); tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    tmp.replace(p)
def fasta(p):
    lines=Path(p).read_text().splitlines(); require(sum(l.startswith('>') for l in lines)==1,'Expected single FASTA record')
    return ''.join(l.strip() for l in lines if not l.startswith('>')).upper()
def split_change(d):
    left=(abs(d)//2)*(1 if d>=0 else -1);return left,d-left

def build(seq,start,pos,ref,alt):
    i=pos-start; d=len(alt)-len(ref); l,r=split_change(d)
    s=i-W//2; e=i+W//2
    require(0<=i and i+len(ref)<=len(seq),'REF outside reference')
    require(seq[i:i+len(ref)]==ref,'REF mismatch')
    # signed adjustment: insertion removes flanks; deletion adds flanks
    ms=s+l; me=e-r
    require(0<=min(s,ms) and max(e,me)<=len(seq),'Insufficient flanking reference')
    require(ms<=i and i+len(ref)<=me,'Allele outside adjusted window')
    wt=seq[s:e]
    mt=seq[ms:i]+alt+seq[i+len(ref):me]
    old=seq[s:i]+alt+seq[i+len(ref):e]
    require(len(wt)==len(mt)==W,'Incorrect window length')
    require(re.fullmatch('[ACGT]+',wt) and re.fullmatch('[ACGT]+',mt),'Non-ACGT in input window')
    require(mt[i-ms:i-ms+len(alt)]==alt,'ALT not at intended position')
    # independently apply same edit to a larger genomic segment then crop.
    a=min(s,ms); b=max(e,me); extended=seq[a:i]+alt+seq[i+len(ref):b]
    require(extended[ms-a:ms-a+W]==mt,'Independent mutation/crop check failed')
    return wt,mt,old,dict(ref_start=s+start,ref_end=e+start-1,mut_source_start=ms+start,mut_source_end=me+start-1,net_change=d,left_trim=l,right_trim=r,ref_allele_index0=i-s,mut_allele_index0=i-ms)

def self_test():
    import random
    rng=random.Random(2711); seq=''.join(rng.choices('ACGT',k=20000)); p=10001
    cases=[(1,2),(1,3),(2,1),(3,1),(22,1),(1,19),(2,3),(3,2),(11,12),(4,4)]
    for nr,na in cases:
        ref=seq[p-1:p-1+nr];alt=''.join(rng.choices('ACGT',k=na));wt,mt,old,m=build(seq,1,p,ref,alt)
        assert len(old)==W+na-nr and len(wt)==len(mt)==W
        assert wt==seq[p-1-W//2:p-1+W//2]
        assert m['left_trim']+m['right_trim']==na-nr
    for pos,ref in [(p,'N'),(1,seq[0])]:
        try:build(seq,1,pos,ref,'A')
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    print('Synthetic QC: 10 valid + 2 rejection cases passed',flush=True)

def prepare(args):
    seq=fasta(args.reference); start=args.start
    if not args.partial:
        require(start==START and len(seq)==END-START+1,'Expected original padded chrX interval 108435549-108701743')
        require(args.overlap,'Full QC requires the supplied independent UCSC overlap FASTA')
        check=fasta(args.overlap); off=108439746-start
        require(len(check)==257802 and seq[off:off+len(check)]==check,'Existing UCSC reference overlap mismatch')
    rows=read_csv(args.variants); require(len(rows)==57,'Expected 57 variants')
    require(len({tuple(r[k] for k in KEY) for r in rows})==57,'Duplicate variant key')
    require(len({r['variant_id'] for r in rows})==57,'Duplicate variant ID')
    require(sum(int(r['n_cases']) for r in rows)==65,'Expected 65 cases')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    qc=[]; records=[]
    for row in rows:
        vid=row['variant_id']; p=int(row['GRCh38_pos_VCF']); ref=row['REF']; alt=row['ALT']
        require(row['GRCh38_chr']=='chrX' and ref!=alt and (len(ref)!=1 or len(alt)!=1),'Invalid non-SNV')
        require(re.fullmatch('[ACGT]+',ref) and re.fullmatch('[ACGT]+',alt) and max(len(ref),len(alt))<=64,'Invalid allele')
        require(all(math.isfinite(float(row[k])) for k in ('ref_ll','var_ll','Evo2score')),'Invalid historical score')
        i=p-start;require(i>=0 and seq[i:i+len(ref)]==ref,f'{vid}: REF mismatch')
        try: wt,mt,old,meta=build(seq,start,p,ref,alt)
        except ValueError as exc:
            if args.partial and str(exc)=='Insufficient flanking reference':
                qc.append(dict(variant_id=vid,status='REF_MATCH_FLANK_UNAVAILABLE',ref_match=True));continue
            raise
        records.append(dict(row=row,wt=wt,mt=mt,old=old,meta=meta))
        qc.append(dict(variant_id=vid,status='PASS',ref_match=True))
    write_csv(out/'QC.csv',qc)
    if records:
        write_csv(out/'windows.csv',[dict(variant_id=x['row']['variant_id'],**x['meta'],ref_len=W,mut_len=W,ref_sha256=sha(x['wt']),mut_sha256=sha(x['mt'])) for x in records])
        with (out/'input_sequences.fasta').open('w') as f:
            for x in records:
                for kind in ('wt','mt'):f.write(f">{x['row']['variant_id']}_{kind}\n{x[kind]}\n")
    manifest=dict(partial=args.partial,reference_start=start,reference_end=start+len(seq)-1,reference_sha256=sha(seq),records=records)
    (out/'prepared.json').write_text(json.dumps(manifest,indent=2))
    print(f'QC: REF 57/57; complete windows {len(records)}/57; {out}',flush=True)

def file_sha(p):
    with open(p,'rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def cache_check(hub):
    hub=Path(hub); repo=hub/'models--arcinstitute--evo2_7b'
    main=repo/'refs/main'; require(main.is_file(),'Missing cached refs/main')
    rev=main.read_text().strip(); require(rev==EXPECTED_REV,f'Cached main revision mismatch: {rev}')
    snap=repo/'snapshots'/rev; require(snap.is_dir(),'Missing expected snapshot')
    require((snap/'config.json').is_file(),'Missing cached config.json')
    merged=hub.parent/(MODEL+'.pt'); full=snap/(MODEL+'.pt')
    # Prefer the exact snapshot's complete weight if available; otherwise compare merged file to cached shards.
    if merged.is_file():
        weights=merged; provenance='existing merged weight; revision inferred from cache and historical ref/old mutant checks'
        if full.is_file():
            require(merged.stat().st_size==full.stat().st_size and (os.path.samefile(merged,full) or file_sha(merged)==file_sha(full)),'Merged weight differs from expected snapshot')
            provenance='merged weight SHA256 matches expected snapshot complete weight'
    elif full.is_file():
        weights=full; provenance='complete weight in expected snapshot'
    else: raise ValueError('No complete cached weight; do not download or merge on paid VM')
    require(weights.stat().st_size>10_000_000_000,'Cached weight unexpectedly small')
    parts=sorted(snap.glob(MODEL+'.pt.part*'),key=lambda x:int(x.name.rsplit('part',1)[1]))
    if weights==merged and parts and all(x.is_file() for x in parts) and sum(x.stat().st_size for x in parts)==merged.stat().st_size and [int(x.name.rsplit('part',1)[1]) for x in parts]==list(range(len(parts))):
        h=hashlib.sha256()
        for part in parts:
            with part.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        require(h.hexdigest()==file_sha(merged),'Merged checkpoint differs from snapshot shards')
        provenance='merged checkpoint SHA256 matches concatenated cached snapshot shards'
    return dict(cache_root=str(hub),cache_revision=rev,snapshot=str(snap),weights=str(weights),weight_bytes=weights.stat().st_size,weight_provenance=provenance)

def diff_status(diff,allow):
    if abs(diff)<=1e-5:return 'PASS'
    if abs(diff)<=1e-4 and allow:return 'WARN'
    return 'FAIL'

def validate_prepared(out, manifest):
    rec=manifest['records']; require(not manifest['partial'] and len(rec)==57,'GPU scoring requires full QC')
    require(len({tuple(x['row'][k] for k in KEY) for x in rec})==57,'Invalid prepared keys')
    require(sum(int(x['row']['n_cases']) for x in rec)==65,'Invalid prepared case count')
    checks={r['variant_id']:r for r in read_csv(out/'windows.csv')}
    require(len(checks)==57,'Invalid windows manifest')
    for x in rec:
        r=x['row']; c=checks[r['variant_id']]; m=x['meta']
        require(len(x['wt'])==len(x['mt'])==W and len(x['old'])==W+m['net_change'],'Invalid prepared length')
        require(sha(x['wt'])==c['ref_sha256'] and sha(x['mt'])==c['mut_sha256'],'Prepared window checksum mismatch')
        i=m['ref_allele_index0']; j=m['mut_allele_index0']; ref=r['REF']; alt=r['ALT']
        require(x['wt'][i:i+len(ref)]==ref and x['mt'][j:j+len(alt)]==alt,'Prepared allele mismatch')
        require(x['old']==x['wt'][:i]+alt+x['wt'][i+len(ref):],'Old mutant mismatch')

def score(args):
    t=time.monotonic();out=Path(args.out); manifest=json.loads((out/'prepared.json').read_text()); records=manifest['records']
    validate_prepared(out,manifest)
    require(not any((out/f).exists() for f in ('runtime.json','reference_check.csv','scores_PARTIAL.csv','scores_COMPLETE.csv')),'Existing run files present: preserve them and use a new output directory')
    require(os.environ.get('HF_HUB_OFFLINE')=='1','Set HF_HUB_OFFLINE=1 before launching Python')
    from huggingface_hub import constants
    cached=cache_check(constants.HF_HUB_CACHE)
    print(cached,flush=True)
    require(args.minutes>0,'Positive time budget required')
    import importlib.metadata as md, inspect
    import torch
    from evo2.models import Evo2
    versions={k:md.version(k) for k in ('evo2','vtx')}
    require(versions=={'evo2':'0.6.0','vtx':'1.1.0'},f'Unexpected package versions: {versions}')
    require(torch.cuda.is_available(),'CUDA unavailable')
    sig=inspect.signature(Evo2.score_sequences)
    for k,v in dict(batch_size=1,prepend_bos=False,reduce_method='mean',average_reverse_complement=False).items():
        require(k in sig.parameters and sig.parameters[k].default==v,f'Unexpected scoring default: {k}')
    ctor=inspect.signature(Evo2)
    require('use_kernels' in ctor.parameters and ctor.parameters['use_kernels'].default is False,'Unexpected kernel setting')
    import evo2.models as em
    require(getattr(em,'HAS_TE',None) is False,'Expected historical bf16 fallback (HAS_TE=False); review before running')
    loader=inspect.getsource(Evo2.load_evo2_model)
    require(all(x in loader for x in ('HF_HUB_CACHE','final_weights_path','snapshot_download','hf_hub_download')), 'Unexpected loader cache logic: review before running')
    from evo2.utils import CONFIG_MAP
    require(CONFIG_MAP[MODEL].endswith('evo2-7b-1m.yml'),'Unexpected 7B config path')
    info=dict(model=MODEL,**cached,versions=versions,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),signature=str(sig),ref_tolerance=1e-5,allow_numeric_drift=args.allow_numeric_drift,hard_limit=1e-4 if args.allow_numeric_drift else 1e-5,reference_sha256=manifest['reference_sha256'],reuse_ref=False,HF_HUB_OFFLINE=os.environ['HF_HUB_OFFLINE'],script_sha256=file_sha(Path(__file__)),prepared_sha256=file_sha(out/'prepared.json'),loader_sha256=file_sha(inspect.getfile(Evo2)),ctor_signature=str(ctor),HAS_TE=em.HAS_TE)
    for name,cmd in [('pip_freeze.txt',[sys.executable,'-m','pip','freeze']),('nvidia_smi.txt',['nvidia-smi'])]:
        try:
            cp=subprocess.run(cmd,capture_output=True,text=True,timeout=15); (out/name).write_text(cp.stdout+cp.stderr)
        except (OSError,subprocess.TimeoutExpired) as exc: (out/name).write_text(str(exc))
    (out/'runtime.json').write_text(json.dumps(info,indent=2)); print(info,flush=True)
    # Match the reconstructed historical constructor. The installed loader must
    # use the validated HF cache path; offline mode blocks downloads.
    load_t=time.monotonic()
    model=Evo2(MODEL)
    print(f"Model load: {time.monotonic()-load_t:.1f}s; elapsed: {time.monotonic()-t:.1f}s",flush=True)
    def one(seq):
        require(time.monotonic()-t<args.minutes*60,'Time budget reached; preserve saved outputs and notify user; user will manually deallocate VM')
        one_t=time.monotonic()
        v=float(model.score_sequences([seq])[0])
        print(f"Sequence {len(seq)}bp: {time.monotonic()-one_t:.2f}s; elapsed: {time.monotonic()-t:.1f}s",flush=True)
        require(math.isfinite(v),'Nonfinite model score');return v
    score_prepared(records,out,one,args.allow_numeric_drift)

def score_prepared(records,out,one,allow_numeric_drift):
    # All reference checks and legacy mutant controls must pass before any new mutant is scored.
    cache={}; refs=[]; checked=set()
    def check_ref(x):
        row=x['row']; h=sha(x['wt'])
        if h not in cache:cache[h]=one(x['wt'])
        new=cache[h]; old=float(row['ref_ll']); diff=new-old
        if row['variant_id'] not in checked:
            refs.append(dict(variant_id=row['variant_id'],old_ref_ll=old,new_ref_ll=new,ref_difference=diff,status=diff_status(diff,allow_numeric_drift),ref_sha256=h))
            checked.add(row['variant_id']);write_csv(out/'reference_check.csv',refs)
        status=diff_status(diff,allow_numeric_drift)
        require(status!='FAIL',f"{row['variant_id']}: reference score mismatch {diff}; stop and investigate")
        if status=='WARN': print(f"WARN {row['variant_id']} reference difference {diff}",flush=True)
        return new
    # Preselected controls: minimum, maximum and median signed length change (distinct variants).
    ordered=sorted(records,key=lambda x:(x['meta']['net_change'],x['row']['variant_id']))
    controls=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    require(tuple(x['row']['variant_id'] for x in controls)==CONTROL_IDS and tuple(x['meta']['net_change'] for x in controls)==CONTROL_CHANGES,'Unexpected control selection; stop')
    for x in records:check_ref(x)
    print('All 57 WT reference checks passed (or pre-authorized WARN).',flush=True)
    old_checks=[]
    for x in controls:
        newref=check_ref(x); oldvar=one(x['old']); row=x['row']; diff=oldvar-float(row['var_ll'])
        delta_diff=(oldvar-newref)-float(row['Evo2score'])
        status=diff_status(diff,allow_numeric_drift); delta_status=diff_status(delta_diff,allow_numeric_drift)
        old_checks.append(dict(variant_id=row['variant_id'],net_change=x['meta']['net_change'],old_length=len(x['old']),old_ref_ll=row['ref_ll'],recomputed_ref_ll=newref,old_var_ll=row['var_ll'],recomputed_old_var_ll=oldvar,var_difference=diff,old_delta=row['Evo2score'],recomputed_old_delta=oldvar-newref,delta_difference=delta_diff,old_mutant_sha256=sha(x['old']),status=status,delta_status=delta_status))
        write_csv(out/'old_mutant_controls.csv',old_checks)
        require(status!='FAIL' and delta_status!='FAIL',f"{row['variant_id']}: old mutant control mismatch; stop")
        if status=='WARN' or delta_status=='WARN':print(f"WARN {row['variant_id']} old mutant control",flush=True)
    print('All WT and legacy mutant checks passed (or pre-authorized WARN); begin new mutant scoring',flush=True)
    results=[]
    for x in records:
        row=x['row']; newref=check_ref(x); newvar=one(x['mt']); newdelta=newvar-newref
        result=dict(row,**x['meta'],new_ref_ll=newref,new_var_ll=newvar,new_Evo2score=newdelta,delta_change=newdelta-float(row['Evo2score']),ref_sha256=sha(x['wt']),mut_sha256=sha(x['mt']))
        results.append(result);write_csv(out/'scores_PARTIAL.csv',results)
        print(f"{len(results)}/57 {row['variant_id']} delta={newdelta:.9g}",flush=True)
    write_csv(out/'scores_COMPLETE.csv',results)
    print('COMPLETE 57/57; save outputs and notify user. User will manually deallocate VM; AI must not stop VM.',flush=True)

def post_qc(source,dest):
    rows=read_csv(source); require(len(rows)==57,'Expected 57 completed scores'); result=[]
    for r in rows:
        n=abs(int(r['net_change'])); diff=abs(float(r['delta_change'])); limit=0.0002 if n<=5 else (0.0015 if n>=9 else None)
        result.append(dict(variant_id=r['variant_id'],absolute_net_change=n,absolute_delta_change=diff,review_threshold=limit if limit is not None else '',status='REVIEW' if limit is not None and diff>limit else ('WITHIN_GUIDE' if limit is not None else 'NO_GUIDE')))
    write_csv(Path(dest),result)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('test'); a=sub.add_parser('prepare'); a.add_argument('--variants',required=True);a.add_argument('--reference',required=True);a.add_argument('--start',type=int,default=START);a.add_argument('--overlap');a.add_argument('--out',required=True);a.add_argument('--partial',action='store_true')
    b=sub.add_parser('score');b.add_argument('--out',required=True);b.add_argument('--allow-numeric-drift',action='store_true');b.add_argument('--minutes',type=float,default=35)
    c=sub.add_parser('post-qc');c.add_argument('--scores',required=True);c.add_argument('--out',required=True)
    args=p.parse_args(); {'test':lambda:self_test(),'prepare':lambda:prepare(args),'score':lambda:score(args),'post-qc':lambda:post_qc(args.scores,args.out)}[args.cmd]()
if __name__=='__main__':main()
