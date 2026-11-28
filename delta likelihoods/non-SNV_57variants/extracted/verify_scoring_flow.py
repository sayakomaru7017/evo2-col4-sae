"""CPU-only mocked scorer: test control order and fail-closed behavior, not GPU accuracy."""
import importlib.util, json, tempfile, contextlib, io
from pathlib import Path
B=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rescore', B/'02_XLAS_nonSNV_rescore.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=json.loads((B/'rescore_20261128_final/prepared.json').read_text())['records']
byid={x['row']['variant_id']:x for x in r}
controls=[byid[k] for k in m.CONTROL_IDS]
expected=[x['wt'] for x in r]+[x['old'] for x in controls]+[x['mt'] for x in r]
values=[float(x['row']['ref_ll']) for x in r]+[float(x['row']['var_ll']) for x in controls]+[float(x['row']['var_ll'])+0.0002 for x in r]
def run(fault=None, drift=0, allow=False, fail=False):
    calls=[]
    def one(seq):
        i=len(calls);assert seq==expected[i],f'wrong order at {i}';calls.append(i)
        return values[i]+(drift if i==fault else 0)
    with tempfile.TemporaryDirectory() as t:
        out=Path(t)
        with contextlib.redirect_stdout(io.StringIO()):
            try:m.score_prepared(r,out,one,allow)
            except ValueError:
                assert fail
            else:assert not fail
        if fail:
            assert len(calls)==fault+1 and not (out/'scores_PARTIAL.csv').exists() and not (out/'scores_COMPLETE.csv').exists()
        else:
            assert len(calls)==117
            assert len(m.read_csv(out/'scores_COMPLETE.csv'))==57
            c=m.read_csv(out/'old_mutant_controls.csv');assert [x['variant_id'] for x in c]==list(m.CONTROL_IDS)
            assert all('recomputed_old_delta' in x and 'var_difference' in x for x in c)
run();run(0,2e-5,fail=True);run(56,2e-5,fail=True);run(57,2e-5,fail=True);run(59,2e-5,fail=True);run(57,2e-5,allow=True);run(57,2e-4,allow=True,fail=True)
print('7/7 mocked scoring-flow tests passed; actual GPU scoring NOT tested.')
