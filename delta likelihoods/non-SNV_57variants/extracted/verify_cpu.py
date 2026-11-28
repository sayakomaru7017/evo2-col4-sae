#!/usr/bin/env python3
"""Verify the provided reference pickle without executing pickle opcodes, then
compare prepared windows against the uploaded reconstructed legacy function.
CPU-only; requires Python standard library. No model or network access.
"""
import ast, importlib.util, json, pickletools, hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rescore',B/'02_XLAS_nonSNV_rescore.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=B/'col4a5_padded_sequence.pkl'; ops=list(pickletools.genops(p.read_bytes()))
assert [op.name for op,arg,pos in ops]==['PROTO','BINUNICODE','FRAME','MEMOIZE','BININT','TUPLE2','MEMOIZE','STOP'], 'Unexpected pickle structure; do not unpickle automatically'
seq,start=ops[1][1],ops[4][1];assert start==m.START and len(seq)==m.END-m.START+1; assert set(seq)==set('ACGT')
assert m.fasta(B/'col4a5_padded_sequence.fasta')==seq
out=B/'rescore_20261128_final'; manifest=json.loads((out/'prepared.json').read_text());m.validate_prepared(out,manifest);assert manifest['reference_sha256']==m.sha(seq)
legacy=B/'run_teshigyou_kakunin_evo2_score_AS_ACTUALLY_EXECUTED.py';tree=ast.parse(legacy.read_text());fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_window');env={'WINDOW_SIZE':8192}
# Execute ONLY the provided, reviewed window slicing function, never main().
exec(compile(ast.Module(body=[fun],type_ignores=[]),str(legacy),'exec'),env)
records=manifest['records']; rows=m.read_csv(B/'XLAS_nonSNV_57_variants.csv');assert [x['row'] for x in records]==rows
for x in records:
    r=x['row']; args=(seq,start,int(r['GRCh38_pos_VCF']),r['REF'],r['ALT'])
    w,o=env['build_window'](*args); assert (w,o)==(x['wt'],x['old']),r['variant_id']
    assert m.build(*args)[:3]==(w,x['mt'],o),r['variant_id']
ordered=sorted(records,key=lambda x:(x['meta']['net_change'],x['row']['variant_id']));controls=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
summary=dict(reference_start=start,reference_end=start+len(seq)-1,reference_length=len(seq),reference_sha256=m.sha(seq),pkl_sha256=m.file_sha(p),legacy_script_sha256=m.file_sha(legacy),REF_match=57,complete_fixed_windows=57,legacy_reference_exact_match=57,legacy_mutant_exact_match=57,unique_reference_windows=len({x['wt'] for x in records}),old_mutant_length_range=[min(len(x['old']) for x in records),max(len(x['old']) for x in records)],max_allele_length=max(max(len(x['row']['REF']),len(x['row']['ALT'])) for x in records),max_old_delta_rounding_error=max(abs(float(x['row']['var_ll'])-float(x['row']['ref_ll'])-float(x['row']['Evo2score'])) for x in records),controls=[dict(variant_id=x['row']['variant_id'],net_change=x['meta']['net_change']) for x in controls],planned_GPU_sequence_calls=len({x['wt'] for x in records})+len(records)+len(controls),GPU_tested=False)
print(json.dumps(summary,indent=2));(B/'CPU_VERIFICATION.json').write_text(json.dumps(summary,indent=2)+'\n')
