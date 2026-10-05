"""
probe.py -- the gradient-scale probe.

Diagnostic only. For every loss term it measures how hard that term pushes the
parameters it is supposed to train, in the units of the Adam-based tuning
rule, and says whether its weight looks too weak or too strong. It never
changes a weight, never stops a run, never touches an optimizer.

=======================================================================
WHAT IS MEASURED
=======================================================================
For a term k with UNWEIGHTED value L_k and declared parameter set P_k
(PROBE_SETS below):

    norm_k = sqrt( sum_{p in P_k}  lr_p^2 * || dL_k / dp ||^2 )

i.e. the root-sum-square of the LR-scaled gradients (lr_p is the learning
rate of the optimizer group that owns p, read from the optimizers). Because
the weighted gradient is w_k * dL_k/dp, the weighted norm is w_k * norm_k:
linear in the weight. The tuning rule is

    scaler_k(w) = target / (w * norm_k)          target = 0.01
    tune w until the scaler is about 1,  i.e.  w*_k = target / norm_k.

Reported per term: norm, current weight, target weight, ramp fraction, the
scaler AT THE TARGET WEIGHT (a ramping term is smaller by design, so the
ramp fraction is shown next to it), the suggested weight w*, and a flag:
    "strong"  scaler < band_lo  (the weight is too large for the rule)
    "weak"    scaler > band_hi  (too small)
    "no_gradient"  the term gives exactly zero gradient on its parameters
    "target_zero"  the target weight is 0, so no scaler (w* still shown)
Flags only; the band (default 0.1 to 10) is loose.

=======================================================================
HOW IT IS COMPUTED
=======================================================================
One forward pass of the network per batch (z sampled, as in training); then,
per term, torch.autograd.grad of the unweighted value on the declared
parameters with retain_graph=True. torch.autograd.grad does NOT write .grad,
so the probe cannot interfere with a training step (no zeroing is needed, and
it is safe to call at any epoch boundary). Terms are computed whether or not
their current weight is 0. Norms of several batches are averaged.

Side effects that are undone: the torch RNG state (CPU and the batch's CUDA
device) is restored, the model's train/eval mode is restored, and BILIP's
recorded first value is restored (the probe's evaluation of BILIP must not
become the "first value computed with weight > 0").

Terms that are skipped (norm None, with the reason in "skipped"):
    sparse  no unresolved block (all nodes frozen)
    moral   no reference adjacency yet (before the first sink check)
    inj     no geodesic sub-table supplied
    bilip   not due this epoch (include_bilip=False)

=======================================================================
UNVALIDATED ROWS
=======================================================================
SPARSE and MORAL act only on the adjacency, whose optimizer is SGD with
momentum, where the step really is LR times the gradient, whereas the 0.01
constant was calibrated with Adam. Their rows carry "adj_only": True; read
their scalers with that in mind.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ----------------------------------------------------------------------
# Declared parameter sets per term.  UNCONFIRMED: edit here if wrong.
#   enc   encoder side of HEAD (AutoencoderKL encoder, quant_conv, pre_linear)
#   dec   decoder side of HEAD (AutoencoderKL decoder, post_quant_conv, post_linear)
#   prior everything in M1Prior except the adjacency
#   adj   the adjacency matrix
# ----------------------------------------------------------------------
PROBE_SETS: Dict[str, Tuple[str, ...]] = {
    "rec": ("enc", "dec"),
    "kl": ("enc", "prior", "adj"),
    "sparse": ("adj",),
    "moral": ("adj",),
    "inj": ("enc",),
    "bilip": ("enc", "dec"),  # decoder directly, encoder through mu
}

# parameter-name prefixes (names as seen from M1Model.named_parameters())
_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "enc": ("net.head.ae.encoder.", "net.head.ae.quant_conv.", "net.head.pre_linear."),
    "dec": ("net.head.ae.decoder.", "net.head.ae.post_quant_conv.", "net.head.post_linear."),
    "adj": ("net.prior.adj_mat.",),
}


def param_groups(model) -> Dict[str, List[torch.nn.Parameter]]:
    """Partition model.parameters() into enc / dec / prior / adj. Raises
    ValueError listing any parameter that fits no group (so a renamed module,
    e.g. after a diffusers update, is caught instead of silently dropped)."""
    groups: Dict[str, List[torch.nn.Parameter]] = {k: [] for k in ("enc", "dec", "prior", "adj")}
    unmapped: List[str] = []
    for n, p in model.named_parameters():
        for g, pre in _PREFIXES.items():
            if n.startswith(pre):
                groups[g].append(p)
                break
        else:
            if n.startswith("net.prior."):
                groups["prior"].append(p)
            else:
                unmapped.append(n)
    if unmapped:
        raise ValueError(f"probe: parameters in no group: {unmapped[:8]}"
                         + (" ..." if len(unmapped) > 8 else ""))
    if len(groups["adj"]) != 1:
        raise ValueError(f"probe: expected exactly one adjacency parameter, got {len(groups['adj'])}")
    return groups


def assess_term(norm: float, target_weight: float, pcfg) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """(scaler at the target weight, suggested weight, flag) for one term.
    Pure python. pcfg needs .target, .band_lo, .band_hi."""
    suggested = (pcfg.target / norm) if norm > 0 else None
    if target_weight == 0:
        return None, suggested, "target_zero"
    if norm == 0:
        return None, None, "no_gradient"
    scaler = pcfg.target / (target_weight * norm)
    flag = "weak" if scaler > pcfg.band_hi else ("strong" if scaler < pcfg.band_lo else None)
    return scaler, suggested, flag


def _term_fns(model, out, x, geod_sub, include_bilip):
    """Per batch: {term: callable giving the unweighted value} and
    {term: reason it is skipped}."""
    T = model.losses.terms
    fns, skip = {}, {}
    fns["rec"] = lambda: T["rec"](out, x)
    fns["kl"] = lambda: T["kl"](out)
    if out["current_adj"].shape[0] == 0:
        skip["sparse"] = "no unresolved block"
    else:
        fns["sparse"] = lambda: T["sparse"](out)
    if T["moral"].prev_adj is None:
        skip["moral"] = "no reference adjacency yet"
    else:
        fns["moral"] = lambda: T["moral"](out)
    if geod_sub is None:
        skip["inj"] = "no geodesic table"
    else:
        fns["inj"] = lambda: T["inj"](out, geod_sub)
    if not include_bilip:
        skip["bilip"] = "not due this epoch"
    else:
        fns["bilip"] = lambda: T["bilip"](out, model.net.head.decode)
    return fns, skip


def grad_scale_probe(
    model,
    batches: Sequence[Tuple[torch.Tensor, ...]],
    lr_by_param: Dict[int, float],
    include_bilip: bool = True,
) -> Dict[str, dict]:
    """
    Args
        model        M1Model (uses model.net, model.losses, model.run_cfg.probe)
        batches      list of (x, y, t, e, geod_sub_or_None), tensors already on
                     the model's device
        lr_by_param  {id(param): lr of its optimizer group}
        include_bilip  probe the (expensive) BILIP term this time
    Returns {term: dict} in model.losses.ORDER with keys: norm (None if
    skipped), weight, target, ramp_fraction, scaler_at_target,
    suggested_weight, flag, adj_only, skipped (reason or None), n_batches.
    """
    pcfg = model.run_cfg.probe
    T = model.losses.terms
    groups = param_groups(model)
    declared = {k: [p for g in gs for p in groups[g]] for k, gs in PROBE_SETS.items()}
    for ps in declared.values():
        for p in ps:
            if id(p) not in lr_by_param:
                raise KeyError("probe: a declared parameter is in no optimizer group")

    norms: Dict[str, List[float]] = {k: [] for k in model.losses.ORDER}
    skipped: Dict[str, str] = {}
    was_training = model.training
    bilip_first = T["bilip"].first_value
    dev = batches[0][0].device
    fork_devices = ([dev.index if dev.index is not None else torch.cuda.current_device()]
                    if dev.type == "cuda" else [])
    try:
        with torch.random.fork_rng(devices=fork_devices), torch.enable_grad():
            model.train()
            for x, y, t, e, geod_sub in batches:
                out = model.net(x, y, t, e, sample=True)
                fns, skip = _term_fns(model, out, x, geod_sub, include_bilip)
                skipped.update(skip)
                for name, fn in fns.items():
                    loss = fn()
                    params = declared[name]
                    if not loss.requires_grad:
                        norms[name].append(0.0)
                        continue
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    sq = 0.0
                    for p, g in zip(params, grads):
                        if g is not None:
                            sq += float((lr_by_param[id(p)] * g).pow(2).sum())
                    norms[name].append(sq ** 0.5)
    finally:
        model.train(was_training)
        T["bilip"].first_value = bilip_first

    res: Dict[str, dict] = {}
    for name in model.losses.ORDER:
        term = T[name]
        row = {
            "norm": None, "weight": term.weight(), "target": term.target,
            "ramp_fraction": term.fraction(), "scaler_at_target": None,
            "suggested_weight": None, "flag": None,
            "adj_only": PROBE_SETS[name] == ("adj",),
            "skipped": None, "n_batches": len(norms[name]),
        }
        if not norms[name]:
            row["skipped"] = skipped.get(name, "not computed")
        else:
            nrm = sum(norms[name]) / len(norms[name])
            row["norm"] = nrm
            row["scaler_at_target"], row["suggested_weight"], row["flag"] = assess_term(
                nrm, term.target, pcfg)
        res[name] = row
    return res
