"""
Compute Evo2 SAE feature activations for COL4A1-6 (GRCh38).

Pipeline
--------
1. Fetch + strand sequences from Ensembl REST API (GRCh38, 1-based inclusive).
2. Reverse-complement minus-strand genes.
3. Load evo2_7b_262k + Goodfire SAE (layer 26, expansion x8, top-k 64).
4. For sequences <= MAX_CHUNK bp: single forward pass.
   For longer sequences: overlapping or non-overlapping chunks (see get_features).
5. Save sparse (COO) feature matrices as .npz under SAE/features/.

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_col4_sae.py
"""

import os, sys, time, pickle, requests
import numpy as np
import torch
from typing import List, Optional, Callable
from huggingface_hub import hf_hub_download
from evo2 import Evo2

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_grad_enabled(False)

# ── Paths ────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
SEQ_PKL  = os.path.join(HERE, "col4_sequences.pkl")   # sequence cache
OUT_DIR  = os.path.join(HERE, "features")
os.makedirs(OUT_DIR, exist_ok=True)

SAE_LAYER   = "blocks-26"
# A100 80GB: evo2_7b_262k uses ~67GB for 158K bp (model 15GB + activations ~52GB).
# Activation memory scales roughly linearly with sequence length.
# 130K bp → ~15 + 0.327*130 ≈ 57GB, safely under 80GB.
MAX_CHUNK   = 130_000

# ── Gene table (GRCh38, 1-based inclusive, Ensembl convention) ───────────────
GENES = {
    "COL4A1": {"chrom": "13", "start": 110_148_963, "end": 110_307_202, "strand": "-"},
    "COL4A2": {"chrom": "13", "start": 110_305_812, "end": 110_513_209, "strand": "+"},
    "COL4A3": {"chrom":  "2", "start": 227_164_624, "end": 227_314_792, "strand": "+"},
    "COL4A4": {"chrom":  "2", "start": 226_967_360, "end": 227_164_488, "strand": "-"},
    "COL4A5": {"chrom":  "X", "start": 108_439_745, "end": 108_697_547, "strand": "+"},
    "COL4A6": {"chrom":  "X", "start": 108_155_607, "end": 108_439_497, "strand": "-"},
}

GENE_ORDER = ["COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6"]

_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ── Sequence fetching ─────────────────────────────────────────────────────────
def fetch_ensembl(chrom: str, start: int, end: int, retries: int = 5) -> str:
    """Return the forward-strand sequence from Ensembl GRCh38 REST API."""
    url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{start}..{end}:1"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"Content-Type": "text/plain"}, timeout=180)
            r.raise_for_status()
            seq = r.text.strip()
            assert len(seq) == (end - start + 1), \
                f"Expected {end-start+1} bp, got {len(seq)}"
            return seq
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [{chrom}:{start}-{end}] attempt {attempt+1} failed: {exc}; "
                  f"retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {chrom}:{start}-{end} after {retries} attempts")


def load_or_fetch_sequences() -> dict:
    if os.path.exists(SEQ_PKL):
        print(f"Loading cached sequences from {SEQ_PKL}")
        return pickle.load(open(SEQ_PKL, "rb"))

    print("Fetching GRCh38 sequences from Ensembl REST API …")
    seqs = {}
    for gene, g in GENES.items():
        chrom, start, end, strand = g["chrom"], g["start"], g["end"], g["strand"]
        region = f"chr{chrom}:{start:,}-{end:,}"
        print(f"  {gene}  {region}  {strand}  ({end-start+1:,} bp)")
        raw = fetch_ensembl(chrom, start, end)
        seq = revcomp(raw) if strand == "-" else raw
        seqs[gene] = {
            "seq": seq, "region": region,
            "chrom": f"chr{chrom}", "start": start, "end": end, "strand": strand,
        }
        time.sleep(0.5)   # be polite to the API

    pickle.dump(seqs, open(SEQ_PKL, "wb"))
    print(f"  Saved sequence cache to {SEQ_PKL}\n")
    return seqs


# ── Observable Evo2 (forward-pass hook infrastructure) ───────────────────────
class ModelScope:
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self._module_dict = {}
        self._build_module_dict()

    def _build_module_dict(self):
        def recurse(module, prefix=""):
            for name, child in module.named_children():
                self._module_dict[prefix + name] = child
                recurse(child, prefix + name + "-")
        recurse(self.model)

    def add_hook(self, hook_fn, module_str, hook_name):
        self.hooks[hook_name] = self._module_dict[module_str].register_forward_hook(hook_fn)

    def remove_all_hooks(self):
        for h in list(self.hooks.values()):
            h.remove()
        self.hooks.clear()


class ObservableEvo2:
    def __init__(self, model_name: str):
        self.evo_model  = Evo2(model_name)
        self.scope      = ModelScope(self.evo_model.model)
        self.tokenizer  = self.evo_model.tokenizer
        self.model      = self.evo_model.model
        self.d_hidden   = 4096

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def forward(self, toks, cache_at: List[str]):
        cache = {}
        for layer in cache_at:
            def _hook(mod, inp, out, layer=layer):
                acts = out[0] if isinstance(out, tuple) else out
                cache[layer] = acts.detach()
                return out
            self.scope.add_hook(_hook, layer, f"cache-{layer}")
        try:
            self.model(toks)
            return {l: a.clone() for l, a in cache.items()}
        finally:
            self.scope.remove_all_hooks()


# ── SAE ───────────────────────────────────────────────────────────────────────
class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, k, device, dtype):
        super().__init__()
        self.d_in, self.d_hidden, self.k = d_in, d_hidden, k
        W = torch.randn(d_in, d_hidden)
        W = 0.1 * W / torch.linalg.norm(W, dim=0, keepdim=True)
        self.W     = torch.nn.Parameter(W)
        self.b_enc = torch.nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        self.to(device, dtype)

    def encode(self, x):
        from math import prod
        f = torch.nn.functional.relu(x @ self.W + self.b_enc)
        numel = self.k * prod(x.shape[:-1])
        top   = torch.topk(f.flatten(), numel)
        return torch.zeros_like(f.flatten()).scatter(-1, top.indices, top.values).reshape(f.shape)

    def forward(self, x):
        f = self.encode(x)
        return f @ self.W.T + self.b_dec, f


def enable_hcl_kernel(evo2_model: ObservableEvo2):
    """
    Patch HyenaCascade layers to use the tiled _hcl_compute_filter kernel.

    By default, compute_filter builds an intermediate (num_poles, D, L) tensor which
    OOMs on an 80 GB A100 for L ≥ ~130 K.  With use_hcl_kernel=True the vortex
    ops kernel builds h tile-by-tile and returns only (1, D, L), avoiding the spike.
    """
    try:
        from vortex.model.model import HyenaCascade
        from vortex.ops.hcl_interface import _hcl_compute_filter  # noqa: F401
    except ImportError as e:
        print(f"  WARNING: could not import HCL kernel ({e}); long sequences may OOM.")
        return
    count = 0
    for module in evo2_model.model.modules():
        if isinstance(module, HyenaCascade):
            module.engine.use_hcl_kernel = True
            count += 1
    print(f"  Enabled tiled HCL kernel on {count} HyenaCascade layers "
          f"(avoids the (num_poles, D, L) intermediate that OOMs for L > 130 K).")


def load_sae(sae_path, d_hidden, device, dtype, expansion=8):
    state = torch.load(sae_path, weights_only=True, map_location="cpu")
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    sae   = BatchTopKTiedSAE(d_hidden, d_hidden * expansion, 64, device, dtype)
    sae.load_state_dict(state)
    return sae


# ── Feature extraction ────────────────────────────────────────────────────────
def _encode_seq(seq: str, evo: ObservableEvo2, sae: BatchTopKTiedSAE) -> np.ndarray:
    """Single forward pass; returns float32 numpy (L, n_features)."""
    toks = torch.tensor(evo.tokenizer.tokenize(seq), dtype=torch.long
                        ).unsqueeze(0).to(evo.device)
    acts = evo.forward(toks, cache_at=[SAE_LAYER])
    feats = sae.encode(acts[SAE_LAYER][0])          # (L, n_features)
    return feats.cpu().float().numpy()


def get_features(seq: str, evo: ObservableEvo2, sae: BatchTopKTiedSAE) -> np.ndarray:
    """Handle sequences longer than MAX_CHUNK.

    For L <= 2*MAX_CHUNK: two overlapping chunks stitched at the midpoint so
    each half has ~MAX_CHUNK/2 bp of upstream context.

    For L > 2*MAX_CHUNK (COL4A6 ~284 kb): non-overlapping chunks of MAX_CHUNK;
    context is lost at chunk boundaries but the run stays within A100 memory.
    """
    L = len(seq)
    if L <= MAX_CHUNK:
        return _encode_seq(seq, evo, sae)

    if L <= 2 * MAX_CHUNK:
        # Two overlapping chunks; stitch at midpoint
        mid          = L // 2
        chunk2_start = L - MAX_CHUNK      # chunk2 covers [chunk2_start : L]
        print(f"    {L:,} bp — two-chunk stitch at midpoint {mid:,}")
        f1 = _encode_seq(seq[:MAX_CHUNK],    evo, sae)
        torch.cuda.empty_cache()
        f2 = _encode_seq(seq[chunk2_start:], evo, sae)
        torch.cuda.empty_cache()
        # offset >= 0 because L <= 2*MAX_CHUNK ⟹ mid <= MAX_CHUNK and
        #   chunk2_start = L - MAX_CHUNK <= MAX_CHUNK ⟹ offset = mid - chunk2_start >= 0
        offset = mid - chunk2_start
        result = np.concatenate([f1[:mid], f2[offset:]], axis=0)
    else:
        # Three or more non-overlapping MAX_CHUNK slices (only COL4A6 hits this)
        n_chunks = (L + MAX_CHUNK - 1) // MAX_CHUNK
        print(f"    {L:,} bp — {n_chunks} non-overlapping chunks of {MAX_CHUNK:,}")
        pieces = []
        for i in range(n_chunks):
            s, e = i * MAX_CHUNK, min((i + 1) * MAX_CHUNK, L)
            print(f"      chunk {i+1}/{n_chunks}: [{s:,}:{e:,}]")
            pieces.append(_encode_seq(seq[s:e], evo, sae))
            torch.cuda.empty_cache()
        result = np.concatenate(pieces, axis=0)

    assert result.shape[0] == L, f"Stitch error: {result.shape[0]} != {L}"
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Sequences
    seqs = load_or_fetch_sequences()

    # 2. Model
    print("\nLoading evo2_7b_262k …")
    t0  = time.time()
    evo = ObservableEvo2("evo2_7b_262k")
    print(f"  ready in {time.time()-t0:.1f}s  device={evo.device}  dtype={evo.dtype}")
    enable_hcl_kernel(evo)

    # 3. SAE
    sae_path = hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
        repo_type="model",
    )
    sae = load_sae(sae_path, d_hidden=evo.d_hidden, device=evo.device,
                   dtype=torch.bfloat16, expansion=8)
    n_features = evo.d_hidden * 8
    print(f"  SAE loaded: {n_features} features, top-k 64, layer {SAE_LAYER}\n")

    # 4. Compute features per gene
    manifest = []
    for gene in GENE_ORDER:
        meta = seqs[gene]
        seq  = meta["seq"]
        print(f"=== {gene}  {meta['region']}  {meta['strand']}  {len(seq):,} bp ===")
        t0 = time.time()

        feats = get_features(seq, evo, sae)   # (L, n_features)

        dt   = time.time() - t0
        L, F = feats.shape
        rows, cols = np.nonzero(feats)
        vals = feats[rows, cols].astype(np.float32)

        out = os.path.join(OUT_DIR, f"{gene}_sae_features.npz")
        np.savez_compressed(
            out,
            rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
            shape=np.array([L, F], dtype=np.int64),
            gene=gene, region=meta["region"],
            chrom=meta["chrom"], start=meta["start"], end=meta["end"],
            strand=meta["strand"], seq_len=L, n_features=F,
            topk=64, layer=SAE_LAYER,
        )
        nnz  = len(vals)
        size = os.path.getsize(out) / 1e6
        print(f"  {L:,} x {F:,}  nnz={nnz:,} ({nnz/L:.1f}/pos)  "
              f"{dt:.1f}s  -> {os.path.basename(out)}  [{size:.1f} MB]\n")
        manifest.append(dict(gene=gene, region=meta["region"],
                             seq_len=L, n_features=F, nnz=nnz,
                             npz=os.path.basename(out)))
        del feats
        torch.cuda.empty_cache()

    # 5. Manifest
    try:
        import pandas as pd
        mf = pd.DataFrame(manifest)
        mf.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
        print("=== Manifest ===\n" + mf.to_string(index=False))
    except ImportError:
        for row in manifest:
            print(row)

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
