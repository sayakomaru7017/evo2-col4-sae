#!/usr/bin/env python3
"""CPU only. Export exact interval from a trusted local FASTA or fetch NCBI reference.
No pickle is loaded automatically (structure and trust must be checked separately).
"""
import argparse, pathlib, subprocess, sys, urllib.request, re
START, END=108435549,108701743
BASE=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--reference',type=pathlib.Path);p.add_argument('--out',default='rescore_20261128_final');a=p.parse_args()
if a.reference is None and (BASE/'col4a5_padded_sequence.fasta').is_file():
    a.reference=BASE/'col4a5_padded_sequence.fasta'
if a.reference is None:
    url=f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000023.11&seq_start={START}&seq_stop={END}&strand=1&rettype=fasta&retmode=text'
    req=urllib.request.Request(url,headers={'User-Agent':'XLAS-reference-QC/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r: raw=r.read().decode('ascii')
    lines=raw.splitlines(); assert lines and lines[0].startswith('>NC_000023.11:'), 'Unexpected NCBI accession/header'
    assert f'{START}-{END}' in lines[0], 'Unexpected reference interval'
    seq=''.join(lines[1:]).upper(); assert len(seq)==END-START+1 and re.fullmatch('[ACGTN]+',seq),'Unexpected reference sequence'
    ref=BASE/'col4a5_padded_sequence_NCBI.fasta'
    with ref.open('x') as f:f.write(raw)
else: ref=a.reference.resolve()
subprocess.run([sys.executable,str(BASE/'02_XLAS_nonSNV_rescore.py'),'test'],check=True)
subprocess.run([sys.executable,str(BASE/'02_XLAS_nonSNV_rescore.py'),'prepare','--variants',str(BASE/'XLAS_nonSNV_57_variants.csv'),'--reference',str(ref),'--overlap',str(BASE/'COL4A5_UCSC_overlap.fasta'),'--out',str(pathlib.Path(a.out).resolve())],check=True)
print('CPU preparation complete. Transfer the complete directory AND the revised scoring script to A100.')
