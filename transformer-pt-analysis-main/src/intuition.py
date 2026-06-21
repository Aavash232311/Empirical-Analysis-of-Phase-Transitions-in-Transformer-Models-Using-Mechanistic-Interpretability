
from __future__ import annotations

import math
import torch

"""
Why would we want to do a DFT over c-axis (output)

The model will learn to use the trig identity instead of using the lookup table.

"""

def phi_field(L: torch.Tensor, f_star: int) -> torch.Tensor:
    """
    Compute phi(a,b) = sum_c L_centered(a,b,c) * exp(-2 pi i f* c / p) DFT in output c axis
 
    L      : FloatTensor [p, p, p], axes [a, b, c] -- raw logit table
    f_star : target class frequency (int)
 
    Returns
    -------
    phi : ComplexTensor [p, p]
    """
    p = L.shape[-1]
    assert L.shape == (p, p, p), "L must be cubic [p, p, p]"
 
    # remove the per-(a,b) softmax gauge freedom before any FFT
    L_centered = L - L.mean(dim=-1, keepdim=True)
 
    c = torch.arange(p, device=L.device, dtype=L_centered.dtype)
    phase = torch.exp(-2j * torch.pi * f_star * c / p)              # [p], complex
 
    phi = (L_centered.to(torch.complex64) * phase).sum(dim=-1)      # [p, p], complex
    return phi

