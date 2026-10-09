"""
config.py -- every tunable number of the M1 training run, in ONE place.

No default lives anywhere else: losses.py, m1net.py and model.py import
their config classes from here and hard-code nothing. All values below are
starting guesses unless a comment says otherwise; fix them from the first
runs.

=======================================================================
THE TREE
=======================================================================
  RunConfig
    loss      LossConfig     target weights, per-loss ramps, REC plateau test,
                             KL settle test, BLAE hyper-parameters
    gate      GateConfig     sink-check gate (mode, check_epoch, W/cap table)
    schedule  ScheduleRules  the global, cross-loss rules (caps, run control)
                             and the global on/off switch
    probe     ProbeConfig    gradient-scale probe (how often, band, constant)

Each consumer receives only its own sub-config.

=======================================================================
ONE SWITCH, TWO DERIVED FIELDS
=======================================================================
schedule.enabled is the single global switch. Two other fields follow from
it and are set by RunConfig.finalize(), whatever the user wrote there:
    loss.schedule = schedule.enabled
    gate.mode     = "stable" if schedule.enabled else "clock"
So "disabled" means: every loss uses its target weight from epoch 1, the
adjacency is lower-triangular from epoch 1, the sink check runs on the fixed
clock (every gate.check_epoch epochs), there are no caps and no fine-tune
phase, and the run stops only at max_epochs. (The probe is independent of
this switch.)

=======================================================================
HOW VALUES ARRIVE (main.py)
=======================================================================
    cfg = RunConfig()                          # defaults (this file)
    cfg = RunConfig.load("run.json", base=cfg) # optional JSON file
    cfg = cfg.with_overrides(["loss.lambda_kl=2", "schedule.kl_cap=40"])
    cfg = cfg.finalize()                       # derived fields + validation
The finalized tree is saved WHOLE in the checkpoint (cfg.to_dict()), so a
run is reproducible from the checkpoint alone. Dotted override values are
parsed by the type of the existing field (bool: true/false/on/off/1/0;
tuples as JSON, e.g. gate.table=[[3,3,30],[5,5,50],[1000000000,8,80]]).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, List, Optional, Tuple


# ======================================================================
# Losses
# ======================================================================
@dataclass
class LossConfig:
    """
    Targets and per-loss mechanics.

    Target weights (0 = never computed):
      lambda_kl, lambda_rec, lambda_sparse, lambda_moral, lambda_inj,
      lambda_bilip
    Per-loss mechanics (used only when scheduling is enabled):
      kl_init_frac        KL weight before its ramp starts, fraction of target
      kl_ramp_epochs, sparse_ramp_epochs, moral_ramp_epochs, bilip_ramp_epochs
      plateau_window, plateau_tol    REC plateau test (REC owns the detector)
      settle_window, settle_tol      KL settle test (KL owns the detector):
                          the (validation, else train) epoch-mean KL has
                          changed by less than settle_tol (relative) between
                          settle_window epochs ago and now, counting from
                          the last reset (the switch of the adjacency to
                          lower-triangular). KL is noisier than REC, hence
                          separate numbers.
    schedule              DERIVED from ScheduleRules.enabled by finalize().
    BLAE hyper-parameters:
      inj_thresh          lower ratio bound of the injective loss (0.3)
      bilip_L             upper bound on diag of the pulled-back metric (2.0)
      bilip_subset_frac   fraction of batch for the Jacobian (0.3)
    Values from Ng's toy code: lambda_kl=1, lambda_rec=10, lambda_sparse=0.01.
    lambda_inj=0.1 and lambda_bilip=1e-4 are starting guesses (no published
    anchor); lambda_moral=0 for the first runs.
    """

    lambda_kl: float = 1.0
    lambda_rec: float = 10.0
    lambda_sparse: float = 0.01
    lambda_moral: float = 0.0
    lambda_inj: float = 0.1
    lambda_bilip: float = 1e-4
    kl_init_frac: float = 1e-3
    kl_ramp_epochs: int = 10
    sparse_ramp_epochs: int = 10
    moral_ramp_epochs: int = 10  # placeholder, not fixed yet
    bilip_ramp_epochs: int = 20
    plateau_window: int = 5
    plateau_tol: float = 0.01
    settle_window: int = 5
    settle_tol: float = 0.02
    schedule: bool = True  # derived
    inj_thresh: float = 0.3
    bilip_L: float = 2.0
    bilip_subset_frac: float = 0.3  # BLAE paper text says 10%; unresolved

    def validate(self) -> None:
        for n in ("lambda_kl", "lambda_rec", "lambda_sparse", "lambda_moral",
                  "lambda_inj", "lambda_bilip"):
            if getattr(self, n) < 0:
                raise ValueError(f"{n} must be >= 0.")
        for n in ("kl_ramp_epochs", "sparse_ramp_epochs", "moral_ramp_epochs",
                  "bilip_ramp_epochs"):
            if getattr(self, n) < 0:
                raise ValueError(f"{n} must be >= 0.")
        if not 0.0 <= self.kl_init_frac <= 1.0:
            raise ValueError("kl_init_frac must be in [0, 1].")
        if self.plateau_window < 1 or self.plateau_tol <= 0:
            raise ValueError("need plateau_window >= 1 and plateau_tol > 0.")
        if self.settle_window < 1 or self.settle_tol <= 0:
            raise ValueError("need settle_window >= 1 and settle_tol > 0.")
        if not 0.0 < self.bilip_subset_frac <= 1.0:
            raise ValueError("bilip_subset_frac must be in (0, 1].")


# ======================================================================
# Sink-check gate
# ======================================================================
@dataclass
class GateConfig:
    """
    Numbers of the sink-check gate.

    mode         DERIVED by finalize(): "stable" (scheduling on) | "clock"
                 (scheduling off).
    check_epoch  clock mode: freeze every this many epochs (Ng's value 5).
    table        stable mode: rows (m_max, W, cap), sorted by m_max. For m
                 unfrozen nodes the first row with m <= m_max applies:
                   W    epochs of unchanged binarized block needed to freeze
                   cap  epochs since the last reset after which the run stops
                        (nothing is frozen at the cap).
                 Default: m 1-3 -> (3, 30); m 4-5 -> (5, 50); m >= 6 -> (8, 80).
    """

    mode: str = "stable"  # derived
    check_epoch: int = 5
    table: Tuple[Tuple[int, int, int], ...] = (
        (3, 3, 30),
        (5, 5, 50),
        (10 ** 9, 8, 80),
    )

    def validate(self) -> None:
        if self.mode not in ("stable", "clock"):
            raise ValueError("gate mode must be 'stable' or 'clock'.")
        if self.mode == "clock" and self.check_epoch < 1:
            raise ValueError("gate check_epoch must be >= 1 in clock mode.")
        if not self.table:
            raise ValueError("gate table must have at least one row.")
        last = 0
        for m_max, w, cap in self.table:
            if m_max <= last:
                raise ValueError("gate table rows must have increasing m_max.")
            if w < 1 or cap < w:
                raise ValueError("gate table needs 1 <= W <= cap in every row.")
            last = m_max

    def params(self, m: int) -> Tuple[int, int]:
        """(W, cap) for m unfrozen nodes."""
        for m_max, w, cap in self.table:
            if m <= m_max:
                return w, cap
        return self.table[-1][1], self.table[-1][2]


# ======================================================================
# Global rules
# ======================================================================
@dataclass
class ScheduleRules:
    """
    The global, cross-loss rules (decided by the model; see model.py, one
    method per rule) and the global switch.

    enabled          THE global switch. False = no scheduling (see the
                     module docstring).
    kl_cap           KL waits for the REC plateau at most this many epochs
                     (counted from the start of training). At the cap the
                     model asks REC whether the head is learning at all:
                     REC below rec_frac * baseline -> start KL anyway, with a
                     warning; otherwise stop the run. If the baseline is
                     unknown the model cannot judge: it starts KL with a
                     warning saying so.
    bilip_cap        BILIP waits for the REC plateau at most this many
                     epochs, then starts anyway, with a warning.
    tril_ramp_epochs when KL is ready, the adjacency is blended from the
                     full off-diagonal mask to the lower-triangular one,
                     a = 0 -> 1 linearly over this many epochs (<= 0: switch
                     at once). No jump in the prior, hence none in the
                     gradients Adam has to follow.
    tril_cap         once the blend has reached 1, SPARSE waits for the KL
                     settle test at most this many epochs, then starts
                     anyway, with a warning.
    rec_frac         fraction of the trivial REC baseline used at the KL cap.
    gate_open_frac   the sink-check gate is held off until SPARSE's weight
                     reaches this fraction of its target (quick hack: the
                     adjacency starts all ones and does not change before
                     there is sparsity pressure, so the gate would see a
                     "stable" block and freeze too early; the right value
                     depends on SPARSE's ramp slope and weight).
    max_epochs       hard stop.
    finetune_epochs  epochs to continue after all content nodes are frozen
                     (graph fixed; SPARSE and MORAL off), then stop.
    """

    enabled: bool = True
    kl_cap: int = 30
    bilip_cap: int = 60
    tril_ramp_epochs: int = 10
    tril_cap: int = 60
    rec_frac: float = 0.5
    gate_open_frac: float = 0.5
    max_epochs: int = 250
    finetune_epochs: int = 20

    def validate(self) -> None:
        if self.kl_cap < 1 or self.bilip_cap < 1 or self.tril_cap < 1:
            raise ValueError("kl_cap, bilip_cap and tril_cap must be >= 1.")
        if self.tril_ramp_epochs < 0:
            raise ValueError("tril_ramp_epochs must be >= 0.")
        if not 0.0 < self.rec_frac <= 1.0:
            raise ValueError("rec_frac must be in (0, 1].")
        if not 0.0 <= self.gate_open_frac <= 1.0:
            raise ValueError("gate_open_frac must be in [0, 1].")
        if self.max_epochs < 1 or self.finetune_epochs < 0:
            raise ValueError("need max_epochs >= 1 and finetune_epochs >= 0.")


# ======================================================================
# Gradient-scale probe
# ======================================================================
@dataclass
class ProbeConfig:
    """
    The gradient-scale probe (probe.py, called by the Trainer between
    epochs). Diagnostic only: it never changes a weight and never stops a run.

    enabled      False = never probe.
    every        probe every this many epochs (0 = never).
    bilip_every  BILIP is expensive; it is included only every this many
                 epochs (0 = never probed). Probing happens only at epochs
                 that are also multiples of `every`.
    n_batches    training batches per probe; the per-term norms are averaged.
    target       the constant of the Adam heuristic scaler = target /
                 (LR * ||grad||); 0.01 (the tuning rule: each weight is
                 tuned until its scaler is about 1).
    band_lo, band_hi
                 a term is flagged "strong" if its scaler at the target
                 weight is below band_lo and "weak" if above band_hi
                 (loose band, flags only).
    """

    enabled: bool = True
    every: int = 5
    bilip_every: int = 25
    n_batches: int = 1
    target: float = 0.01
    band_lo: float = 0.1
    band_hi: float = 10.0

    def validate(self) -> None:
        if self.every < 0 or self.bilip_every < 0:
            raise ValueError("probe every / bilip_every must be >= 0.")
        if self.n_batches < 1:
            raise ValueError("probe n_batches must be >= 1.")
        if self.target <= 0:
            raise ValueError("probe target must be > 0.")
        if not 0.0 < self.band_lo < self.band_hi:
            raise ValueError("probe needs 0 < band_lo < band_hi.")


# ======================================================================
# The tree
# ======================================================================
@dataclass
class RunConfig:
    loss: LossConfig = field(default_factory=LossConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    schedule: ScheduleRules = field(default_factory=ScheduleRules)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    # ---------------- derived fields + validation ----------------
    def finalize(self) -> "RunConfig":
        """Copy with the derived fields set from schedule.enabled, validated."""
        c = copy.deepcopy(self)
        c.loss.schedule = c.schedule.enabled
        c.gate.mode = "stable" if c.schedule.enabled else "clock"
        c.validate()
        return c

    def validate(self) -> None:
        self.loss.validate()
        self.gate.validate()
        self.schedule.validate()
        self.probe.validate()

    # ---------------- (de)serialisation ----------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict, base: Optional["RunConfig"] = None) -> "RunConfig":
        """Build from a (possibly partial) nested dict, on top of `base`
        (default: all defaults). Unknown keys raise KeyError."""
        cfg = copy.deepcopy(base) if base is not None else cls()
        _update(cfg, d)
        return cfg

    @classmethod
    def load(cls, path: str, base: Optional["RunConfig"] = None) -> "RunConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f), base)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def with_overrides(self, items: List[str]) -> "RunConfig":
        """Apply dotted overrides, e.g. ["loss.lambda_kl=2", "schedule.enabled=off"]."""
        cfg = copy.deepcopy(self)
        for it in items:
            if "=" not in it:
                raise ValueError(f"override must look like section.field=value, got {it!r}")
            path, val = it.split("=", 1)
            parts = path.strip().split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = _get(obj, p)
            name = parts[-1]
            setattr(obj, name, _coerce(_get(obj, name), val))
        return cfg


# ======================================================================
# helpers
# ======================================================================
def _get(obj: Any, name: str) -> Any:
    if not hasattr(obj, name):
        raise KeyError(f"unknown config field {name!r} in {type(obj).__name__}")
    return getattr(obj, name)


def _tupleize(v: Any) -> Any:
    return tuple(_tupleize(x) for x in v) if isinstance(v, (list, tuple)) else v


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"cannot read {v!r} as a boolean")


def _coerce(cur: Any, v: Any) -> Any:
    """Convert v to the type of the existing value `cur`."""
    if isinstance(cur, bool):  # before int: bool is an int subclass
        return _to_bool(v)
    if isinstance(cur, int):
        return int(v)
    if isinstance(cur, float):
        return float(v)
    if isinstance(cur, tuple):
        return _tupleize(json.loads(v) if isinstance(v, str) else v)
    return v


def _update(obj: Any, d: dict) -> None:
    for k, v in d.items():
        cur = _get(obj, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise TypeError(f"{k!r} must be a dict of fields")
            _update(cur, v)
        else:
            setattr(obj, k, _coerce(cur, v))


# ======================================================================
# Smoke test (no torch): python config.py
# ======================================================================
if __name__ == "__main__":
    c = RunConfig().with_overrides(
        ["loss.lambda_kl=2", "schedule.kl_cap=40", "schedule.enabled=off",
         "gate.table=[[3,2,20],[1000000000,4,40]]", "probe.every=3", "probe.enabled=off"]).finalize()
    assert c.loss.lambda_kl == 2.0 and c.schedule.kl_cap == 40
    assert c.loss.schedule is False and c.gate.mode == "clock"
    assert c.gate.params(2) == (2, 20) and c.gate.params(9) == (4, 40)
    assert c.probe.every == 3 and c.probe.enabled is False
    c2 = RunConfig.from_dict(json.loads(json.dumps(c.to_dict()))).finalize()
    assert c2.to_dict() == c.to_dict()
    assert RunConfig().finalize().gate.mode == "stable"
    c3 = RunConfig().with_overrides(["loss.settle_window=7", "loss.settle_tol=0.05",
                                     "schedule.tril_cap=33"]).finalize()
    assert c3.loss.settle_window == 7 and c3.loss.settle_tol == 0.05
    assert c3.schedule.tril_cap == 33
    assert RunConfig().finalize().schedule.tril_ramp_epochs == 10
    assert RunConfig().with_overrides(["schedule.tril_ramp_epochs=0"]).finalize().schedule.tril_ramp_epochs == 0
    for bad in (["nope.x=1"], ["loss.nope=1"], ["schedule.rec_frac=2"],
                ["probe.band_lo=20"], ["probe.n_batches=0"], ["probe.target=0"],
                ["loss.settle_window=0"], ["loss.settle_tol=0"],
                ["schedule.tril_cap=0"], ["schedule.tril_ramp_epochs=-1"]):
        try:
            RunConfig().with_overrides(bad).finalize()
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError(bad)
    print("config ok")
