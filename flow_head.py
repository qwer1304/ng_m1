"""Generic invertible ("flow") head for M1.

======================================================================
WHAT THIS MODULE IS
======================================================================
The trainable part of the flow-based estimator of sec:flow. The full
pipeline is

    x  --BB (frozen, cached)-->  x_tilde  --Head.encode-->  z_hat

where
  * BB is the pretrained, bijective i-RevNet backbone, run ONCE per image
    and cached. This module does NOT contain BB. Use the `out_bij` output
    of i-RevNet (never `out`, which is the lossy classifier path).
  * x_tilde is the tensor BB produces at the cut point you chose, shape
    (B, C, H, W). BB is volume preserving, so log|det J_BB| = 0 exactly
    and no BB term ever appears in the loss.
  * Head is an exactly invertible network built here. It maps x_tilde to
    the latent z_hat over which the M1 prior (content ANM prior on
    c=(y,t), style prior on r=(t,e)) is defined. The M1 prior, the
    adjacency matrix, the sparsity penalty and sink freezing live
    OUTSIDE this module; the Head only provides z_hat and a log-det.

======================================================================
DIRECTION CONVENTION (easy to get backwards)
======================================================================
The note writes z_hat = head^{-1}(x_tilde). Here the network is built
directly in the ENCODING direction:

    z_hat, logdet_enc = head.encode(x_tilde)     # x_tilde -> z_hat
    x_tilde           = head.decode(z_hat)       # exact inverse

logdet_enc = log|det d z_hat / d x_tilde|, one value per sample, shape (B,).
The term log|det J_head(z_hat)| of eq. (flow-loss) equals -logdet_enc.
So the per-sample training loss is

    loss = -log p_prior(z_hat | u) - logdet_enc + lambda_sparse * l_sparse

(the BB term is 0). Encoding is deterministic: no sampling, no
mu/logvar, no KL term, no reconstruction term.

======================================================================
ARCHITECTURE
======================================================================
Head = sequence of layers, built from a schedule (`scales`, see ScaleSpec):

    [Squeeze?] FlowStep x n_blocks   [Squeeze?] FlowStep x n_blocks  ...

FlowStep = ActNorm2d -> InvConv1x1LU -> AffineCoupling (Glow-style):
  * ActNorm2d     per-channel affine, data-dependent initialisation.
  * InvConv1x1LU  learned invertible channel mixing, LU parameterised so
                  the log-det is a sum over a diagonal (cheap).
  * AffineCoupling  y1 = x1 ; y2 = x2 * exp(s(x1)) + t(x1), where s, t
                  come from a small conv net that never has to be
                  inverted. log-det = sum of s.
Squeeze = space-to-depth, (C,H,W) -> (4C,H/2,W/2). Pure reshape, log-det 0.

The 1x1 conv mixes all channels in every step, so using a FIXED channel
split in the coupling is fine (no alternating masks needed).

Deliberately NOT done: Glow's multi-scale "factor out half the channels"
step. z_hat must keep all elements of x_tilde because the M1 prior is
defined over the full z_hat.

======================================================================
CHOOSING THE BB CUT POINT (in_channels, in_h, in_w)
======================================================================
BB (i-RevNet 301: nBlocks=[6,16,72,6], nChannels=[24,96,384,1536], input
3x224x224) has four stages; channels below count both streams together.
Every cut keeps all 150,528 numbers per image.

    cut after stage 1:   48 x 56 x 56   -> needs 3 squeezes to reach 7x7
    cut after stage 2:  192 x 28 x 28   -> needs 2 squeezes
    cut after stage 3:  768 x 14 x 14   -> needs 1 squeeze
    cut after stage 4: 3072 x  7 x  7   -> needs 0 squeezes

Nothing forces you to squeeze all the way down; the schedule is yours.
Earlier cut = features closer to pixels (style more locally visible) but
the head does more work on large maps (more compute and memory).
Later cut = cheap, but features are arranged by the classification
objective. Which cut works best is an empirical question (see the
calibration loop of sec:flow:depth): compare held-out NLL and a
content/style separability probe across cuts.

======================================================================
PREDICTION AT AN UNSEEN LOCATION e* (e.g. L100)
======================================================================
Prediction uses encode() only. decode() is NOT part of it. Follows
sec:use; the prior itself is outside this module.

  1. x_tilde = BB(x*) (cached), then z_hat, _ = head.encode(x_tilde).
  2. Split z_hat into content z_c and style z_s (the split is defined by
     the estimated graph: style = isolated nodes whose distribution
     depends on (t,e) but not on y). z_s is not used below.
  3a. Time of day t* observed:
          p(y | x*, t*, e*)  ~  p^{e*}(y | t*) * p(z_c | y, t*)
      Evaluate the content prior p(z_c | y, t*) (mean_net(adj * z_c),
      scale from c=(y,t*)) for every species y, multiply by the label
      prior p^{e*}(y | t*) (pooled training prior, EM on unlabeled
      images of e*, or uniform), normalise over y, take argmax.
      The location-specific style parameters of e* are NOT needed, and
      logdet_enc is NOT needed: it depends on x* only, not on y, so it
      cancels in the normalisation over y.
  3b. t* not observed:
          p(y | x*, e*)  ~  sum_t p^{e*}(y,t) * p(z_c | y,t)
      (style dropped: correct posterior given z_c alone, but not the full
      one). Optionally use brightness b as a soft proxy for t:
          sum_t p(y,t) p(z_c | y,t) p(b | t), with p(b | t) from pooled
      training data (see sec:use).

What decode() is for: (i) checking invertibility (reconstruction error);
(ii) counterfactuals, e.g. replace z_s by style values of another
location, decode, and inspect the resulting x_tilde. Both are
diagnostics, not part of the M1 pipeline.

OPEN ITEM: this module returns z_hat with C*H*W coordinates, while the
M1 model has n_c + n_s latents (a handful). How the prior and the
content/style split are defined over such a z_hat is decided outside
this file and is not settled here.

======================================================================
ASSUMPTIONS
======================================================================
  * x_tilde comes from an exactly invertible BB, so the composite
    BB o g is a diffeomorphism and Ng et al.'s theory still applies.
    If BB is lossy, nothing here repairs that.
  * Input spatial size is even wherever a Squeeze is applied (asserted).
  * Input is float32 (the 1x1 conv inverse uses torch.inverse).
  * ActNorm data-dependent init needs a batch of >= 2 samples.
  * Pixel/feature scale of x_tilde is arbitrary: ActNorm normalises it.

======================================================================
LIMITATIONS / COSTS
======================================================================
  * The 1x1 conv stores three CxC matrices per step (l, u, and the
    permutation-free factors; only about half of l and u is used). At
    C=3072 that is ~28M parameters PER STEP; 20 steps is ~570M before
    the coupling subnets. Use fewer steps at wide levels, or an earlier
    cut where C is smaller.
  * decode() inverts a CxC matrix per step (torch.inverse): slow. Neither
    training nor prediction needs it (both use encode only); it is a
    diagnostic tool (see PREDICTION below), so the cost rarely matters.
  * Affine coupling can locally rescale density (BB cannot), so the head
    is the ONLY place that reshapes density. An all-additive head would
    leave all density shaping to the prior.
  * Whether a finite-depth head fits well is empirical; nothing here
    guarantees observational equivalence.
  * Checkpoints: ActNorm holds an `initialized` buffer. When loading a
    trained checkpoint, construct with actnorm_data_init=True or False
    as you like, then load_state_dict; the buffer restores the flag. To
    avoid re-initialising on the first forward pass of a fresh model
    you want to keep, set actnorm_data_init=False.
  * This file was not executed in the environment where it was written
    (PyTorch was unavailable there). Run `python flow_head.py` first: it
    should print the output shape and a reconstruction error around 1e-5
    or smaller.

======================================================================
EXAMPLE
======================================================================
Cut after stage 2 (192 x 28 x 28), two squeezes down to 3072 x 7 x 7:

    scales = [
        ScaleSpec(squeeze_before=False, n_blocks=4, hidden_channels=128,
                  kernel_size=3, keep_ratio=0.5),   # 192 x 28 x 28
        ScaleSpec(squeeze_before=True,  n_blocks=4, hidden_channels=256,
                  kernel_size=3, keep_ratio=0.5),   # 768 x 14 x 14
        ScaleSpec(squeeze_before=True,  n_blocks=4, hidden_channels=512,
                  kernel_size=3, keep_ratio=0.5),   # 3072 x 7 x 7
    ]
    head = Head(in_channels=192, in_h=28, in_w=28, scales=scales,
                scale_clamp=2.0, actnorm_data_init=True, actnorm_eps=1e-6)
    print(head.out_shape)                   # (3072, 7, 7)

    z_hat, logdet_enc = head.encode(x_tilde)    # x_tilde: (B,192,28,28)
    nll = -prior_log_prob(z_hat, u) - logdet_enc   # (B,) per-sample
    x_back = head.decode(z_hat)                    # exact inverse

Cut after stage 4 (no squeeze, everything at 3072 x 7 x 7):

    scales = [ScaleSpec(False, 10, 512, 3, 0.5)]
    head = Head(3072, 7, 7, scales, 2.0, True, 1e-6)

All hyperparameters are supplied by the caller; there are no defaults.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActNorm2d(nn.Module):
    """Per-channel affine layer, y = (x + loc) * exp(log_scale).

    Invertible for any parameters. log-det per sample = H*W*sum(log_scale)
    (same for every sample, returned as a scalar tensor).

    With data_init=True, the first forward call sets loc and log_scale so
    the output has zero mean and unit variance per channel on that batch.
    That call must use a representative batch of size >= 2 (std is
    computed over batch and spatial positions). Subsequent calls do not
    re-initialise (state kept in the `initialized` buffer).

    Args:
        channels:  number of channels C of the input (B, C, H, W).
        data_init: if True, initialise from the first batch; if False,
                   start at the identity (loc=0, log_scale=0).
        eps:       added to the std before the log, avoids log(0) for
                   constant channels. Small, e.g. 1e-6.
    """

    def __init__(self, channels, data_init, eps):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.data_init = data_init
        self.eps = eps
        self.register_buffer("initialized", torch.tensor(0 if data_init else 1, dtype=torch.uint8))

    @torch.no_grad()
    def _init(self, x):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True)
        self.loc.data.copy_(-mean)
        self.log_scale.data.copy_(-torch.log(std + self.eps))
        self.initialized.fill_(1)

    def forward(self, x):
        """x: (B,C,H,W) -> (y, logdet) with logdet a scalar tensor."""
        if self.initialized.item() == 0:
            self._init(x)
        h, w = x.shape[2:]
        y = (x + self.loc) * torch.exp(self.log_scale)
        logdet = h * w * self.log_scale.sum()
        return y, logdet

    def inverse(self, y):
        """Exact inverse of forward (log-det not needed)."""
        return y * torch.exp(-self.log_scale) - self.loc


class InvConv1x1LU(nn.Module):
    """Learned invertible 1x1 convolution (channel mixing), LU parameterised.

    The C x C weight is W = P @ (L + I) @ (U + diag(sign * exp(log_s))),
    with P a fixed permutation, L strictly lower triangular, U strictly
    upper triangular. Initialised from a random rotation (orthogonal), so
    the initial log-det is 0. log-det per sample = H*W*sum(log_s).

    Cost: three CxC parameter matrices; forward is one 1x1 conv; inverse()
    calls torch.inverse on a CxC matrix (slow for large C, use for
    prediction only).

    Args:
        channels: C, the number of input (= output) channels.
    """

    def __init__(self, channels):
        super().__init__()
        w0 = torch.linalg.qr(torch.randn(channels, channels))[0]  # random rotation
        p, l, u = torch.linalg.lu(w0)
        s = torch.diagonal(u)
        self.register_buffer("p", p)
        self.register_buffer("sign_s", torch.sign(s))
        self.register_buffer("l_mask", torch.tril(torch.ones(channels, channels), -1))
        self.register_buffer("u_mask", torch.triu(torch.ones(channels, channels), 1))
        self.register_buffer("eye", torch.eye(channels))
        self.l = nn.Parameter(l)
        self.u = nn.Parameter(torch.triu(u, 1))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))

    def _weight(self):
        l = self.l * self.l_mask + self.eye
        u = self.u * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.p @ l @ u

    def forward(self, x):
        """x: (B,C,H,W) -> (y, logdet) with logdet a scalar tensor."""
        h, w = x.shape[2:]
        wt = self._weight()
        y = F.conv2d(x, wt.view(*wt.shape, 1, 1))
        return y, h * w * self.log_s.sum()

    def inverse(self, y):
        """Exact inverse of forward (uses torch.inverse; float32 assumed)."""
        wt_inv = torch.inverse(self._weight())
        return F.conv2d(y, wt_inv.view(*wt_inv.shape, 1, 1))


class AffineCoupling(nn.Module):
    """Affine coupling layer.

    Split the channels into x1 (first n_keep) and x2 (the rest):
        y1 = x1
        y2 = x2 * exp(log_s(x1)) + shift(x1)
    log_s and shift come from one small conv net applied to x1. The net
    is only ever run forward, never inverted, so it may be arbitrary.
    log-det per sample = sum of log_s over channels and positions.

    log_s is bounded: log_s = scale_clamp * tanh(raw / scale_clamp), so the
    per-element scale lies in (exp(-scale_clamp), exp(scale_clamp)). This
    keeps early training numerically stable.

    The last conv of the net is zero-initialised, so the layer is exactly
    the identity at initialisation.

    Args:
        channels:        C, total input channels.
        n_keep:          channels passed through unchanged (x1); must
                         satisfy 0 < n_keep < channels. x2 has
                         channels - n_keep channels.
        hidden_channels: width of the subnet's hidden layers.
        kernel_size:     kernel of the first and last conv of the subnet
                         (odd, e.g. 3; the 1x1 middle conv is fixed).
                         Padding is kernel_size // 2, so H, W are kept.
        scale_clamp:     positive float bounding |log_s| (see above).
                         Larger = more expressive, less stable. e.g. 2.0.
    """

    def __init__(self, channels, n_keep, hidden_channels, kernel_size, scale_clamp):
        super().__init__()
        assert 0 < n_keep < channels, "n_keep must be in (0, channels)"
        self.n_keep = n_keep
        self.scale_clamp = scale_clamp
        n_out = channels - n_keep
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(n_keep, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2 * n_out, kernel_size, padding=pad),
        )
        last = self.net[-1]  # zero init -> coupling is identity at start
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _st(self, x1):
        shift, raw = self.net(x1).chunk(2, dim=1)
        log_s = self.scale_clamp * torch.tanh(raw / self.scale_clamp)
        return shift, log_s

    def forward(self, x):
        """x: (B,C,H,W) -> (y, logdet) with logdet of shape (B,)."""
        x1, x2 = x[:, :self.n_keep], x[:, self.n_keep:]
        shift, log_s = self._st(x1)
        y2 = x2 * torch.exp(log_s) + shift
        return torch.cat([x1, y2], dim=1), log_s.flatten(1).sum(1)

    def inverse(self, y):
        """Exact inverse of forward."""
        y1, y2 = y[:, :self.n_keep], y[:, self.n_keep:]
        shift, log_s = self._st(y1)
        x2 = (y2 - shift) * torch.exp(-log_s)
        return torch.cat([y1, x2], dim=1)


class FlowStep(nn.Module):
    """One block: ActNorm2d -> InvConv1x1LU -> AffineCoupling.

    forward returns (y, logdet) with logdet of shape (B,): the sum of the
    three layers' log-dets. inverse undoes the three layers in reverse.

    Args: see ActNorm2d (data_init = actnorm_data_init, eps = actnorm_eps)
    and AffineCoupling (channels, n_keep, hidden_channels, kernel_size,
    scale_clamp).
    """

    def __init__(self, channels, n_keep, hidden_channels, kernel_size,
                 scale_clamp, actnorm_data_init, actnorm_eps):
        super().__init__()
        self.actnorm = ActNorm2d(channels, actnorm_data_init, actnorm_eps)
        self.conv = InvConv1x1LU(channels)
        self.coupling = AffineCoupling(channels, n_keep, hidden_channels,
                                       kernel_size, scale_clamp)

    def forward(self, x):
        x, ld1 = self.actnorm(x)
        x, ld2 = self.conv(x)
        x, ld3 = self.coupling(x)
        return x, ld1 + ld2 + ld3

    def inverse(self, y):
        y = self.coupling.inverse(y)
        y = self.conv.inverse(y)
        return self.actnorm.inverse(y)


class Squeeze(nn.Module):
    """Space-to-depth: (B,C,H,W) -> (B,4C,H/2,W/2).

    Each 2x2 spatial patch becomes 4 extra channels. A pure reshape and
    permutation, so log-det is exactly 0 and no element is lost.
    Requires even H and W (asserted). No parameters.
    """

    def forward(self, x):
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "Squeeze needs even H, W; got %dx%d" % (h, w)
        x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return x.reshape(b, c * 4, h // 2, w // 2), 0.0

    def inverse(self, y):
        """Depth-to-space, exact inverse of forward."""
        b, c4, h, w = y.shape
        y = y.view(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        return y.reshape(b, c4 // 4, h * 2, w * 2)


@dataclass
class ScaleSpec:
    """One resolution level of the head schedule.

    A level = optional Squeeze, then n_blocks FlowSteps at the (possibly
    new) resolution. Channels seen by the steps are the running channel
    count: it is multiplied by 4 (and H, W halved) by every Squeeze.

    Attributes:
        squeeze_before:  if True, apply a Squeeze before this level's
                         blocks (needs even H, W at that point). The
                         first level may also squeeze.
        n_blocks:        number of FlowSteps at this level (>= 0). More
                         blocks = more capacity and more parameters; the
                         note suggests ~10-20 in total, to be tuned.
        hidden_channels: hidden width of every coupling subnet at this
                         level. Wider levels (many channels) usually want
                         larger values; this is the main memory knob.
        kernel_size:     spatial kernel of the coupling subnet (odd int,
                         3 is a sensible choice; use 1 for 1x1 maps).
        keep_ratio:      fraction of the level's channels that stay
                         unchanged and feed the subnet; n_keep =
                         round(C * keep_ratio), which must land in
                         [1, C-1]. 0.5 is the usual choice.
    """
    squeeze_before: bool
    n_blocks: int
    hidden_channels: int
    kernel_size: int
    keep_ratio: float


class Head(nn.Module):
    """Generic multi-scale invertible head, fully controlled by `scales`.

    Args:
        in_channels, in_h, in_w: shape (C, H, W) of x_tilde, i.e. of BB's
            output at the chosen cut point (see the table in the module
            docstring). Fixes the channel counts used when building the
            layers; encode() then expects inputs of exactly this shape.
        scales: list of ScaleSpec, applied in order. This is the whole
            architecture: where to squeeze, and how many blocks/how wide
            at each resolution.
        scale_clamp: positive float, bound on |log_s| in every coupling
            (see AffineCoupling). e.g. 2.0.
        actnorm_data_init: True = every ActNorm initialises from the first
            batch passed to encode() (call it once on a representative
            batch of size >= 2 before training); False = identity init.
        actnorm_eps: small positive float for the ActNorm std, e.g. 1e-6.

    Attributes:
        in_shape:  (C, H, W) of the input.
        out_shape: (C, H, W) of z_hat after all squeezes. The number of
            elements always equals that of the input.

    Methods:
        encode(x_tilde) -> (z_hat, logdet_enc)
            x_tilde: (B, C, H, W) float32.
            z_hat:   (B, *out_shape).
            logdet_enc: (B,), log|det d z_hat / d x_tilde|.
            Use for training and for prediction at L100.
        decode(z_hat) -> x_tilde
            Exact inverse. Slow (matrix inverse per step). Not used in
            training or prediction; diagnostics only (invertibility
            check, counterfactuals). See PREDICTION in the module docs.
        forward(x_tilde) is encode(x_tilde).
    """

    def __init__(self, in_channels, in_h, in_w, scales, scale_clamp,
                 actnorm_data_init, actnorm_eps):
        super().__init__()
        layers = []
        c, h, w = in_channels, in_h, in_w
        for spec in scales:
            if spec.squeeze_before:
                layers.append(Squeeze())
                c, h, w = c * 4, h // 2, w // 2
            n_keep = int(round(c * spec.keep_ratio))
            for _ in range(spec.n_blocks):
                layers.append(FlowStep(c, n_keep, spec.hidden_channels, spec.kernel_size,
                                       scale_clamp, actnorm_data_init, actnorm_eps))
        self.layers = nn.ModuleList(layers)
        self.in_shape = (in_channels, in_h, in_w)
        self.out_shape = (c, h, w)

    def encode(self, x_tilde):
        """x_tilde (B,C,H,W) -> (z_hat, logdet_enc of shape (B,))."""
        z = x_tilde
        logdet = x_tilde.new_zeros(x_tilde.shape[0])
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        return z, logdet

    def decode(self, z):
        """z_hat (B,*out_shape) -> x_tilde, the exact inverse of encode."""
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def forward(self, x_tilde):
        return self.encode(x_tilde)


if __name__ == "__main__":
    # Small invertibility check on a two-level head with one squeeze.
    torch.manual_seed(0)
    scales = [ScaleSpec(False, 2, 32, 3, 0.5), ScaleSpec(True, 2, 32, 3, 0.5)]
    head = Head(in_channels=12, in_h=8, in_w=8, scales=scales, scale_clamp=2.0,
                actnorm_data_init=True, actnorm_eps=1e-6)
    x = torch.randn(4, 12, 8, 8)
    head.encode(x)  # triggers ActNorm data init
    for p in head.parameters():  # perturb so couplings are non-trivial
        p.data.add_(0.01 * torch.randn_like(p))
    z, ld = head.encode(x)
    print("out shape:", head.out_shape, tuple(z.shape[1:]))
    print("max recon err:", (head.decode(z) - x).abs().max().item())
