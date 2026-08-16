"""
Spectral observables for monitoring the grokking phase transition.

All quantities are defined on the pair-state lattice Z_p², which is the
natural state space for the Fibonacci-mod-p prediction task.  Each lattice
point (a, b) corresponds to a state that uniquely determines the next token
y = (a + b) mod p.

References
----------
  Notes: "Monitoring Diagonal Spectral Mass and Structure Factors …" (this repo)
  Nanda et al. (2023) https://arxiv.org/abs/2301.05217

Public API
----------
  monitor_spectral_observables    — one-call interface for training loops
  build_probe_sequences           — canonical input sequences for every (a,b) state
  collect_state_logits            — evaluate model on all p² states
  structure_factor_from_logits    — 2-D power spectrum on Z_p²
  diagonal_spectral_mass          — scalar order parameter in [0, 1]
  dominant_nonzero_mode           — (kx, ky) of the largest non-DC peak
  second_moment_correlation_length — Ornstein–Zernike ξ estimator
"""
from __future__ import annotations

import math
import torch

# ---------------------------------------------------------------------------
# Probe sequences
# ---------------------------------------------------------------------------

def build_probe_sequences(
    n: int, seq_len: int, device: torch.device = None,
    eq_token: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build one probe sequence for every terminal pair (a, b) in Z_n².

    Without eq_token (old format, seq_len=2):
      seqs[i] = [..., a, b]   built by running the Fibonacci recurrence backward.

    With eq_token (Nanda format, seq_len=3):
      seqs[i] = [a, b, eq_token]
      The pair (a, b) sits at positions [-3, -2] and the "=" token is appended
      at position -1.  No backward extension is needed for seq_len=3.

    Parameters
    ----------
    n         : modulus (and number of distinct token values)
    seq_len   : total length of each input sequence
    device    : torch device
    eq_token  : if not None, appended as the last token (Nanda "=" format)

    Returns
    -------
    seqs  : LongTensor [n², seq_len]
    pairs : LongTensor [n², 2]   — the (a, b) pair for each row
    """
    vals  = torch.arange(n, device=device)
    pairs = torch.cartesian_prod(vals, vals)           # [n², 2]
    seqs  = torch.zeros((n * n, seq_len), dtype=torch.long, device=device)

    if eq_token is not None:
        # Nanda format: [..., a, b, =]
        seqs[:, -1] = eq_token
        seqs[:, -3] = pairs[:, 0]
        seqs[:, -2] = pairs[:, 1]
        for i in range(seq_len - 4, -1, -1):
            seqs[:, i] = (seqs[:, i + 2] - seqs[:, i + 1]) % n
    else:
        seqs[:, -2] = pairs[:, 0]
        seqs[:, -1] = pairs[:, 1]
        for i in range(seq_len - 3, -1, -1):
            seqs[:, i] = (seqs[:, i + 2] - seqs[:, i + 1]) % n

    return seqs, pairs


# ---------------------------------------------------------------------------
# Logit table
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_state_logits(
    model: torch.nn.Module,
    n: int,
    seq_len: int,
    device: torch.device,
    batch_size: int = 256,
    eq_token: int = None,
) -> torch.Tensor:
    """
    Evaluate the model on every pair state (a, b) and collect final-position logits.

    The logit table has shape [n, n, n]: axes are [a, b, c] where c ranges over
    the n modular classes 0..n-1.  If the model vocabulary is n+1 (Nanda format
    with an "=" token), the extra class is silently discarded by taking only the
    first n logits at the output position.

    Parameters
    ----------
    eq_token : if not None, passed to build_probe_sequences (Nanda "=" format)

    Returns
    -------
    L : FloatTensor [n, n, n]
    """
    model.eval()
    seqs, pairs = build_probe_sequences(
        n=n, seq_len=seq_len, device=device, eq_token=eq_token
    )
    L = torch.empty((n, n, n), dtype=torch.float32, device=device)

    for start in range(0, seqs.shape[0], batch_size):
        x  = seqs[start : start + batch_size]
        ab = pairs[start : start + batch_size]
        logits = model(x)                        # [B, seq_len, vocab_size]
        L[ab[:, 0], ab[:, 1], :] = logits[:, -1, :n]   # discard eq-token class

    return L


# ---------------------------------------------------------------------------
# Structure factor
# ---------------------------------------------------------------------------

def structure_factor_from_logits(
    L: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the class-summed 2-D structure factor from the logit table L.

    Logits are first centered across classes to remove the softmax gauge
    freedom: adding the same constant to all classes at a state (a, b) does
    not change predictions but would shift the raw Fourier spectrum.

    Parameters
    ----------
    L : FloatTensor [n, n, n]  — axes are [a, b, c]

    Returns
    -------
    S         : FloatTensor [n, n]  — class-summed power spectrum S(k, ℓ)
    L_centered: FloatTensor [n, n, n]
    F         : ComplexTensor [n, n, n]  — 2-D FFT of centered logits
    """
    n = L.shape[0]
    assert L.shape == (n, n, n), "L must be cubic [n, n, n]"

    L_centered = L - L.mean(dim=-1, keepdim=True)
    F = torch.fft.fftn(L_centered, dim=(0, 1), norm="ortho")
    S = F.abs().pow(2).sum(dim=-1)       # sum over class index c
    return S, L_centered, F

# ---------------------------------------------------------------------------
# OZ structure factor
# ---------------------------------------------------------------------------

def oz_sf(
    L: torch.Tensor
):
    """
    Compute the OZ structure factor S(k) at a single target class-frequency f*,
    following eq. (18) of the grokking-transition paper.

    L : logit table, shape (n, n, n) -> (a, b, c)
        a, b index the input pair lattice (Z_p x Z_p)
        c indexes the output class (also mod p)
    f_star : the target class frequency to probe (int, 0 <= f_star < n)

    Returns
    -------
    S       : (n, n) real tensor, the structure factor S(k1, k2)
    phi     : (n, n) complex tensor, the real-space order-parameter field (eq. 18)
    phi_hat : (n, n) complex tensor, the Fourier transform of phi (S = |phi_hat|^2)
    """
    n = L.shape[0]
    assert L.shape == (n, n, n)

    L_centered = L - L.mean(dim=-1, keepdim=True)

    '''
    For every (a,b) pair, run a DFT across all 113 possible frequencies 
    to get a power value at each frequency; then add up each frequency's power
    across all pairs, and whichever frequency has the biggest total is the winner (f*) 
    '''
    F_full = torch.fft.fft(L_centered, dim=-1, norm="ortho")
    power_per_f = F_full.abs().pow(2).sum(dim=(0,1))
    f_star = int(power_per_f.argmax())

    # pick out the single target frequency f* along the class axis c 
    # phi(a,b) = sum_c L_centered(a,b,c) * exp(-2*pi*i*f*_star*c/n)
    phi = F_full[:, :, f_star] * (n ** 0.5)

    # 2D FFT of the order parameter
    phi_hat = torch.fft.fft2(phi, norm="ortho")

    # Structure factor
    S = phi_hat.abs().pow(2)

    return S, phi, phi_hat


def gradient_lat(phi: torch.Tensor):
    """
    Compute the discrete lattice gradient of phi on a 2D grid.

    Args:
        phi: 2D tensor of shape [n, n] representing φ(a,b)

    Returns:
        grad_a: Gradient in the a-direction [n, n]
        grad_b: Gradient in the b-direction [n, n]
    """

    grad_a = torch.zeros_like(phi)
    grad_b = torch.zeros_like(phi)

                    # phi(a + 1, b) - phi(a, b)
    grad_a[:-1, :] = phi[1:, :] - phi[:-1, :] # row below - current row
                    # phi(a, b + 1) - phi(a, b)
    grad_b[:, :-1] = phi[:, 1:] - phi[:, :-1] # col + 1 - current col

    return grad_a, grad_b


def gl_energy(phi: torch.Tensor, r:int, kappa:int, u:int):
    """
    Compute the GL energy of phi on a 2D grid.

    Args:
        phi: 2D tensor of shape [n, n] representing φ(a,b)
        r: constant
        kappa: constant
        u: constant

    Returns:
        energy: GL energy
    """

    grad_a, grad_b = gradient_lat(phi)
    grad_sq = grad_a.abs().pow(2) + grad_b.abs().pow(2)

    energy = (r * phi.abs().pow(2) + kappa * grad_sq + u * phi.abs().pow(4)).sum()
    return energy



# ---------------------------------------------------------------------------
# Scalar order parameters
# ---------------------------------------------------------------------------

def diagonal_spectral_mass(S: torch.Tensor) -> torch.Tensor:
    """
    Fraction of non-DC spectral power that sits on the diagonal k = ℓ.

    The modular-addition rule y = a + b (mod p) decomposes in Fourier space
    exactly onto diagonal modes (k, k), so this quantity rises from ~0
    (random network) to ~1 (fully grokked network) during training.

    Parameters
    ----------
    S : FloatTensor [n, n]  — class-summed structure factor

    Returns
    -------
    M_diag : scalar tensor in [0, 1]
    """
    n   = S.shape[0]
    idx = torch.arange(n, device=S.device)
    dc  = S[0, 0]
    total_ndc = S.sum() - dc
    diag_ndc  = S[idx, idx].sum() - dc
    return diag_ndc / (total_ndc + 1e-12)


def dominant_nonzero_mode(S: torch.Tensor) -> tuple[int, int]:
    """
    Return the lattice indices (kx, ky) of the largest non-DC Fourier mode.

    Parameters
    ----------
    S : FloatTensor [n, n]

    Returns
    -------
    (kx_idx, ky_idx) : int tuple
    """
    n      = S.shape[0]
    S_work = S.clone()
    S_work[0, 0] = -torch.inf
    flat   = torch.argmax(S_work)
    return int(flat // n), int(flat % n)


def second_moment_correlation_length(S: torch.Tensor) -> dict:
    """
    Estimate the Ornstein–Zernike correlation length ξ from the dominant peak.

    Uses the lattice second-moment estimator:
      ξ² = (S_peak / S_nn - 1) / (4 sin²(π/n))
    where S_nn is the mean of the four nearest-neighbour modes on the torus.

    Returns a dict with keys: xi, peak_index, S_peak, S_nn_mean.
    """
    n  = S.shape[0]
    kx, ky   = dominant_nonzero_mode(S)
    S_peak   = S[kx, ky]
    neighbors = [
        ((kx + 1) % n, ky), ((kx - 1) % n, ky),
        (kx, (ky + 1) % n), (kx, (ky - 1) % n),
    ]
    S_nn  = torch.stack([S[i, j] for i, j in neighbors]).mean()
    denom = 4.0 * math.sin(math.pi / n) ** 2
    xi_sq = torch.clamp((S_peak / (S_nn + 1e-12) - 1.0) / (denom + 1e-12), min=0.0)
    return {
        "xi":         xi_sq.sqrt(),
        "peak_index": (kx, ky),
        "S_peak":     S_peak,
        "S_nn_mean":  S_nn,
    }

# ---------------------------------------------------------------------------
# One-call interface for training loops
# ---------------------------------------------------------------------------

@torch.no_grad()
def monitor_spectral_observables(
    model: torch.nn.Module,
    n: int,
    seq_len: int,
    device: torch.device,
    batch_size: int = 256,
    eq_token: int = None,
) -> dict:
    """
    Compute all spectral observables in a single call.

    Intended to be called periodically inside a training loop (e.g., every
    50 epochs) to track the grokking phase transition without breaking the
    training graph.

    Parameters
    ----------
    eq_token : if not None, probe sequences use the Nanda [a, b, =] format
               and the "=" class is excluded from the logit table.

    Returns
    -------
    dict with keys:
      structure_factor       : FloatTensor [n, n]
      diagonal_spectral_mass : float   — scalar order parameter in [0, 1]
      correlation_length     : float   — Ornstein–Zernike ξ estimator
      peak_index             : (int, int)
      S_peak                 : float
      S_nn_mean              : float
    """
    L       = collect_state_logits(model, n=n, seq_len=seq_len,
                                   device=device, batch_size=batch_size,
                                   eq_token=eq_token)
    S, _, _ = structure_factor_from_logits(L)
    m_diag  = diagonal_spectral_mass(S)
    xi_info = second_moment_correlation_length(S)
    return {
        "structure_factor":       S,
        "diagonal_spectral_mass": m_diag.item(),
        "correlation_length":     xi_info["xi"].item(),
        "peak_index":             xi_info["peak_index"],
        "S_peak":                 xi_info["S_peak"].item(),
        "S_nn_mean":              xi_info["S_nn_mean"].item(),
    }

# ---------------------------------------------------------------------------
# LOGISTIC FUNCTION
# ---------------------------------------------------------------------------

def logistic(t, t_g, Delta_t):
    import numpy as np
    """
    Logistic (sigmoid) function for fitting M_diag(t).
    
    Parameters:
        t       : Time (epochs)
        t_g     : Transition midpoint (where M = 0.5)
        Delta_t : Transition width (controls steepness)
    
    Returns:
        M_diag(t) = 1 / (1 + exp(-(t - t_g)/Delta_t))
    """
    return 1.0 / (1.0 + np.exp(-(t - t_g) / Delta_t))