#!/usr/bin/env python3
from __future__ import annotations
import numpy as np


class OscillationProbability:
    """2-flavor νμ → νμ (and ν̄μ → ν̄μ) survival probability for T2K.

    P_surv(E) = 1 - sin²(2θ₂₃) · sin²(1.267 · Δm²₃₂ · L_km / E_GeV)

    T2K best-fit defaults (2024 publication):
        sin2_theta23 = 0.512    →  sin²θ₂₃
        dm2_32       = 2.45e-3  eV²  →  Δm²₃₂

    Latest PDG 2023 θ₁₃:
        sin2_2theta13 = 0.0851  [stored for reference; unused in 2-flavor formula]

    The same formula holds for antineutrinos in the CP-conserving 2-flavor
    approximation (no δ_CP sensitivity). If δ_CP effects become important,
    switch to full 3-flavor PMNS.
    """

    L_KM_T2K: float = 295.0

    def __init__(
        self,
        sin2_theta23: float = 0.512,
        dm2_32: float = 2.45e-3,
        L_km: float = L_KM_T2K,
        sin2_2theta13: float = 0.0851,
    ):
        self.sin2_theta23 = float(sin2_theta23)
        self.dm2_32 = float(dm2_32)
        self.L_km = float(L_km)
        self.sin2_2theta13 = float(sin2_2theta13)

    @property
    def sin2_2theta23(self) -> float:
        return 4.0 * self.sin2_theta23 * (1.0 - self.sin2_theta23)

    @property
    def oscillation_peak_gev(self) -> float:
        """Energy (GeV) at the first oscillation maximum."""
        return 1.267 * self.dm2_32 * self.L_km / (0.5 * np.pi)

    def survival_prob(self, enu_gev: np.ndarray) -> np.ndarray:
        """νμ (or ν̄μ) survival probability at given energies in GeV.

        Vectorised over enu_gev. Returns 1.0 for bins with E ≤ 0.
        """
        enu = np.asarray(enu_gev, dtype=np.float64)
        valid = enu > 0.0
        phase = np.where(valid, 1.267 * self.dm2_32 * self.L_km / np.where(valid, enu, 1.0), 0.0)
        prob = 1.0 - self.sin2_2theta23 * np.sin(phase) ** 2
        return np.where(valid, prob, 1.0)

    def weight_ratio(
        self,
        enu_gev: np.ndarray,
        osc_ref: OscillationProbability,
        min_ref_prob: float = 0.01,
    ) -> np.ndarray:
        """Reweighting factor P_self(E) / P_ref(E).

        Bins where P_ref < min_ref_prob receive weight 1.0 (no reweighting)
        to avoid numerical blow-up near the oscillation minimum.
        """
        p_self = self.survival_prob(enu_gev)
        p_ref = osc_ref.survival_prob(enu_gev)
        safe = p_ref >= min_ref_prob
        ratio = np.where(safe, p_self / np.where(safe, p_ref, 1.0), 1.0)
        return ratio

    def __repr__(self) -> str:
        return (
            f"OscillationProbability("
            f"sin2_theta23={self.sin2_theta23:.4f}, "
            f"dm2_32={self.dm2_32:.4e} eV², "
            f"L={self.L_km:.0f} km, "
            f"peak≈{self.oscillation_peak_gev:.3f} GeV)"
        )
