"""
Compute Evo2 SAE (layer-26, expansion x8, top-k 64 -> 32768 features) feature
activations for COL1A1 (whole gene) and COL4A1-6 (Collagen IV NC1 domain).

Follows evo2/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb:
  - model  : evo2_7b_262k  (the model the SAE was trained on)
  - SAE    : Goodfire/Evo-2-Layer-26-Mixed / sae-layer26-mixed-expansion_8-k_64.pt
  - hook   : blocks-26  ->  encode  ->  (seq_len, 32768) activation timeseries

Sequences (GRCh38) come from col4_col1a1_sequences.pkl (already strand-adjusted:
minus-strand genes reverse-complemented per the Excel 'Evo2 input note').

Output: one sparse .npz per gene (top-k64 => only 64 nonzero per position, so the
COO sparse form is lossless and tiny) under features/<GENE>_sae_features.npz
"""
from typing import List, Optional, Callable
import os, time, pickle
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from evo2 import Evo2

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_grad_enabled(False)

HERE = "/home/azureuser/evo/alport/sae"
SEQ_PKL = os.path.join(HERE, "col4_col1a1_sequences.pkl")
OUT_DIR = os.path.join(HERE, "features")
os.makedirs(OUT_DIR, exist_ok=True)
SAE_LAYER_NAME = "blocks-26"


# ---------------- Notebook machinery (verbatim) ----------------
class ModelScope:
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.activations_cache = {}
        self.override_store = {}
        self._build_module_dict()

    def _build_module_dict(self):
        self._module_dict = {}
        def recurse(module, prefix=''):
            for name, child in module.named_children():
                self._module_dict[prefix + name] = child
                recurse(child, prefix=prefix + name + '-')
        recurse(self.model)

    def list_modules(self):
        return self._module_dict.keys()

    def add_hook(self, hook_fn, module_str, hook_name):
        module = self._module_dict[module_str]
        self.hooks[hook_name] = module.register_forward_hook(hook_fn)

    def remove_hook(self, hook_name):
        self.hooks[hook_name].remove()
        del self.hooks[hook_name]

    def remove_all_hooks(self):
        for hook_name in list(self.hooks.keys()):
            self.remove_hook(hook_name)

    def clear_all_caches(self):
        for k in self.activations_cache.keys():
            self.activations_cache[k] = []


INTERVENTION_INTERFACE = Callable[[torch.Tensor], torch.Tensor]


class ObservableEvo2:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.evo_model = Evo2(model_name)
        self.scope = ModelScope(self.evo_model.model)
        self.tokenizer = self.evo_model.tokenizer
        self.model = self.evo_model.model
        self.d_hidden = 4096

    @property
    def device(self):
        return next(self.evo_model.model.parameters()).device

    @property
    def dtype(self):
        return next(self.evo_model.model.parameters()).dtype

    def forward(self, toks, cache_activations_at: Optional[List[str]] = None,
                interventions: dict = None):
        interventions = interventions or {}
        cache_activations_at = cache_activations_at or []
        output_cache = {}
        layers = list(set(list(interventions.keys()) + cache_activations_at))
        if layers:
            for layer in layers:
                def _intervene(model, inp, output, layer=layer):
                    acts = output[0] if isinstance(output, tuple) else output
                    if layer in interventions:
                        acts = interventions[layer](acts)
                    if layer in cache_activations_at:
                        output_cache[layer] = acts.detach()
                    return (acts, output[1]) if isinstance(output, tuple) else acts
                self.scope.add_hook(_intervene, layer, f'intervene-{layer}')
        try:
            model_outputs = self.model(toks)
            cached = {l: a.clone() for l, a in output_cache.items()}
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()
        return model_outputs[0], cached

    def generate(self, prompt_seqs, n_tokens=1, temperature=1.0, top_k=4, top_p=1.,
                 batched=True, cached_generation=False, verbose=0,
                 cache_activations_at=None, interventions=None):
        interventions = interventions or {}
        cache_activations_at = cache_activations_at or []
        output_cache = {}
        layers = list(set(list(interventions.keys()) + cache_activations_at))
        if layers:
            for layer in layers:
                def _intervene(model, inp, output, layer=layer):
                    acts = output[0]
                    if layer in interventions:
                        acts = interventions[layer](acts)
                    if layer in cache_activations_at and output_cache.get(layer) is None:
                        output_cache[layer] = [acts]
                    elif layer in cache_activations_at:
                        output_cache[layer].append(acts)
                    return (acts, output[1]) if len(output) == 2 else acts
                self.scope.add_hook(_intervene, layer, f'intervene-{layer}')
        try:
            output = self.evo_model.generate(
                prompt_seqs, n_tokens=n_tokens, temperature=temperature, top_k=top_k,
                top_p=top_p, batched=batched, cached_generation=cached_generation, verbose=verbose)
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()
        acts_cache = {l: torch.cat(a, dim=1).clone().detach() for l, a in output_cache.items()}
        return ''.join(output[0]), acts_cache


class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(self, d_in, d_hidden, k, device, dtype, tiebreaker_epsilon=1e-6):
        super().__init__()
        self.d_in, self.d_hidden, self.k = d_in, d_hidden, k
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = torch.nn.Parameter(W_mat)
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_in))
        self.device, self.dtype = device, dtype
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(self.device, self.dtype)

    def encoder_pre(self, x):
        return x @ self.W + self.b_enc

    def encode(self, x, tiebreak=False):
        f = torch.nn.functional.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k, tiebreak=tiebreak)

    def _batch_topk(self, f, k, tiebreak=False):
        from math import prod
        if tiebreak:
            f += self.tiebreaker.broadcast_to(f)
        *input_shape, _ = f.shape
        numel = k * prod(input_shape)
        f_topk = torch.topk(f.flatten(), numel, dim=-1)
        f_topk = torch.zeros_like(f.flatten()).scatter(-1, f_topk.indices, f_topk.values).reshape(f.shape)
        return f_topk

    def decode(self, f):
        return f @ self.W.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def load_topk_sae(sae_path, d_hidden, device, dtype, expansion_factor=16):
    sae_dict = torch.load(sae_path, weights_only=True, map_location="cpu")
    sae_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sae_dict.items()}
    sae = BatchTopKTiedSAE(d_hidden, d_hidden * expansion_factor, 64, device, dtype)
    sae.load_state_dict(sae_dict)
    return sae


# ---------------- Load model + SAE ----------------
print("Loading Evo2-7B-262k (downloads ~weights on first run)...")
t0 = time.time()
model = ObservableEvo2(model_name="evo2_7b_262k")
print(f"  model ready in {time.time()-t0:.1f}s  device={model.device} dtype={model.dtype}")

sae_path = hf_hub_download(repo_id="Goodfire/Evo-2-Layer-26-Mixed",
                           filename="sae-layer26-mixed-expansion_8-k_64.pt", repo_type="model")
topk_sae = load_topk_sae(sae_path, d_hidden=model.d_hidden, device=model.device,
                         dtype=torch.bfloat16, expansion_factor=8)
N_FEATURES = model.d_hidden * 8
print(f"  SAE loaded: {N_FEATURES} features, top-k 64, layer {SAE_LAYER_NAME}")


def get_feature_ts(seq):
    toks = model.tokenizer.tokenize(seq)
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)
    _, acts = model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
    feats = topk_sae.encode(acts[SAE_LAYER_NAME][0])
    return feats.cpu().detach().float().numpy()


def get_feature_ts_via_generate(seq):
    _, acts = model.generate([seq], n_tokens=1, cached_generation=True,
                             cache_activations_at=[SAE_LAYER_NAME])
    feats = topk_sae.encode(acts[SAE_LAYER_NAME][0])
    return feats.cpu().detach().float().numpy()


# ---------------- Compute + save per gene ----------------
seqs = pickle.load(open(SEQ_PKL, "rb"))
order = ["COL1A1", "COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6"]

manifest = []
for gene in order:
    meta = seqs[gene]
    seq = meta["seq"]
    print(f"\n=== {gene} ({meta['region']}, {len(seq)} bp) ===")
    t0 = time.time()
    try:
        feats = get_feature_ts(seq)
        method = "forward"
    except RuntimeError as e:
        print(f"  forward failed ({e}); retrying via generate...")
        torch.cuda.empty_cache()
        feats = get_feature_ts_via_generate(seq)
        method = "generate"
    dt = time.time() - t0
    L, F = feats.shape
    rows, cols = np.nonzero(feats)
    vals = feats[rows, cols].astype(np.float32)
    out = os.path.join(OUT_DIR, f"{gene}_sae_features.npz")
    np.savez_compressed(
        out,
        rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
        shape=np.array([L, F], dtype=np.int64),
        gene=gene, region=meta["region"], chrom=str(meta["chrom"]),
        start=meta["start"], end=meta["end"], strand=meta["strand"],
        seq_len=L, n_features=F, topk=64, layer=SAE_LAYER_NAME, method=method,
    )
    nnz = len(vals)
    print(f"  {L}x{F}  nnz={nnz} ({nnz/L:.1f}/pos)  {dt:.1f}s ({method})  -> {out}"
          f"  [{os.path.getsize(out)/1e6:.1f} MB]")
    manifest.append(dict(gene=gene, region=meta["region"], seq_len=L, n_features=F,
                         nnz=nnz, method=method, npz=os.path.basename(out)))
    del feats
    torch.cuda.empty_cache()

import pandas as pd
mf = pd.DataFrame(manifest)
mf.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
print("\n=== Manifest ===")
print(mf.to_string(index=False))
print(f"\nSaved to {OUT_DIR}")
print(f"Peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
