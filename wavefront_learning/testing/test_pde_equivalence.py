"""Equivalence tests: new PDE interface vs old bias/encoder implementations.

Verifies that LWRPDE, ARZPDE, EulerPDE + PDEBias produce identical outputs
to the original LWRBias, ARZBias, EulerBias and segment encoders.

Run: cd wavefront_learning && uv run python -m pytest testing/test_pde_equivalence.py -v
"""

import torch
import pytest

from models.base.flux import GreenshieldsFlux
from models.base.lwr_bias import LWRBias
from models.base.arz_bias import ARZBias
from models.base.euler_bias import EulerBias, EulerSegmentEncoder
from models.base.characteristic_features import SegmentPhysicsEncoder
from models.base.pde import LWRPDE, ARZPDE, EulerPDE
from models.base.pde_bias import PDEBias


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lwr_ic(B=2, K=4, seed=42):
    """Random LWR IC with realistic values."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(B, K + 1, generator=g).cumsum(dim=1)
    ks = torch.rand(B, K, generator=g)  # rho in [0, 1]
    pieces_mask = torch.ones(B, K)
    return {"xs": xs, "ks": ks, "pieces_mask": pieces_mask}


def _make_arz_ic(B=2, K=4, seed=42):
    """Random ARZ IC with realistic values."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(B, K + 1, generator=g).cumsum(dim=1)
    ks = torch.rand(B, K, generator=g).clamp(min=0.1)  # rho > 0
    ks_v = torch.rand(B, K, generator=g) * 2  # velocity
    pieces_mask = torch.ones(B, K)
    return {"xs": xs, "ks": ks, "ks_v": ks_v, "pieces_mask": pieces_mask}


def _make_euler_ic(B=2, K=4, seed=42):
    """Random Euler IC with realistic values."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.rand(B, K + 1, generator=g).cumsum(dim=1)
    ks = torch.rand(B, K, generator=g).clamp(min=0.1) * 2  # rho > 0
    ks_v = (torch.rand(B, K, generator=g) - 0.5) * 4  # velocity
    ks_p = torch.rand(B, K, generator=g).clamp(min=0.1) * 3  # pressure > 0
    pieces_mask = torch.ones(B, K)
    return {
        "xs": xs,
        "ks": ks,
        "ks_v": ks_v,
        "ks_p": ks_p,
        "pieces_mask": pieces_mask,
    }


def _make_query(B=2, nt=5, nx=8, seed=99):
    """Random query points."""
    g = torch.Generator().manual_seed(seed)
    t_coords = torch.rand(B, nt, nx, generator=g) * 2  # t > 0
    x_coords = torch.rand(B, nt, nx, generator=g) * 4
    return t_coords, x_coords


# ===========================================================================
# LWR equivalence
# ===========================================================================


class TestLWRPhysicsFeatures:
    """LWRPDE.physics_features matches SegmentPhysicsEncoder scalar features."""

    def test_scalar_features_match(self):
        flux = GreenshieldsFlux()
        pde = LWRPDE(flux=flux)
        ic = _make_lwr_ic()
        ks = ic["ks"]

        # New: PDE physics features
        new_feats = pde.physics_features(ic)  # (B, K, 3)

        # Old: what SegmentPhysicsEncoder computes internally
        old_rho = ks
        old_lambda = flux.derivative(ks)
        old_flux = flux(ks)
        old_feats = torch.stack([old_rho, old_lambda, old_flux], dim=-1)

        assert torch.allclose(new_feats, old_feats, atol=1e-6)


class TestLWRBiasEquivalence:
    """PDEBias(LWRPDE) matches LWRBias output."""

    @pytest.mark.parametrize("use_damping", [False, True])
    def test_bias_matches(self, use_damping):
        flux = GreenshieldsFlux()
        sharpness = 5.0

        # Old
        old_bias_mod = LWRBias(
            flux=flux,
            use_damping=use_damping,
            initial_damping_sharpness=sharpness,
        )

        # New
        pde = LWRPDE(flux=flux)
        new_bias_mod = PDEBias(
            pde=pde,
            use_damping=use_damping,
            initial_damping_sharpness=sharpness,
        )

        ic = _make_lwr_ic()
        query = _make_query()

        with torch.no_grad():
            old_out = old_bias_mod(ic, query)
            new_out = new_bias_mod(ic, query)

        assert old_out.shape == new_out.shape
        assert torch.allclose(old_out, new_out, atol=1e-5), (
            f"Max diff: {(old_out - new_out).abs().max().item()}"
        )

    def test_bias_with_masking(self):
        """Padded segments produce identical masking."""
        flux = GreenshieldsFlux()
        ic = _make_lwr_ic(B=1, K=3)
        ic["pieces_mask"] = torch.tensor([[1.0, 1.0, 0.0]])

        old_mod = LWRBias(flux=flux, use_damping=False)
        pde = LWRPDE(flux=flux)
        new_mod = PDEBias(pde=pde, use_damping=False)

        query = _make_query(B=1)

        with torch.no_grad():
            old_out = old_mod(ic, query)
            new_out = new_mod(ic, query)

        assert torch.allclose(old_out, new_out, atol=1e-5)

    @pytest.mark.parametrize("B,nt,nx,K", [(1, 1, 1, 2), (2, 5, 10, 3)])
    def test_output_shape(self, B, nt, nx, K):
        pde = LWRPDE()
        mod = PDEBias(pde=pde, use_damping=False)
        ic = _make_lwr_ic(B=B, K=K)
        query = _make_query(B=B, nt=nt, nx=nx)
        out = mod(ic, query)
        assert out.shape == (B, nt, nx, K)


# ===========================================================================
# ARZ equivalence
# ===========================================================================


class TestARZBiasEquivalence:
    """PDEBias(ARZPDE) matches ARZBias output."""

    @pytest.mark.parametrize("gamma", [1.0, 2.0])
    def test_bias_matches(self, gamma):
        old_mod = ARZBias(gamma=gamma)
        pde = ARZPDE(gamma=gamma)
        new_mod = PDEBias(pde=pde, use_damping=False)

        ic = _make_arz_ic()
        query = _make_query()

        with torch.no_grad():
            old_out = old_mod(ic, query)
            new_out = new_mod(ic, query)

        assert old_out.shape == new_out.shape
        assert torch.allclose(old_out, new_out, atol=1e-5), (
            f"Max diff: {(old_out - new_out).abs().max().item()}"
        )

    def test_bias_with_masking(self):
        ic = _make_arz_ic(B=1, K=3)
        ic["pieces_mask"] = torch.tensor([[1.0, 1.0, 0.0]])

        old_mod = ARZBias(gamma=1.0)
        pde = ARZPDE(gamma=1.0)
        new_mod = PDEBias(pde=pde, use_damping=False)

        query = _make_query(B=1)

        with torch.no_grad():
            old_out = old_mod(ic, query)
            new_out = new_mod(ic, query)

        assert torch.allclose(old_out, new_out, atol=1e-5)


# ===========================================================================
# Euler equivalence
# ===========================================================================


class TestEulerPhysicsFeatures:
    """EulerPDE.physics_features matches EulerSegmentEncoder scalar features."""

    def test_scalar_features_match(self):
        gamma = 1.4
        pde = EulerPDE(gamma=gamma)
        ic = _make_euler_ic()

        new_feats = pde.physics_features(ic)  # (B, K, 7)

        # Old: what EulerSegmentEncoder computes internally
        rho = ic["ks"]
        u = ic["ks_v"]
        p = ic["ks_p"]
        eps = 1e-6
        c = (gamma * p / rho.clamp(min=eps)).clamp(min=eps).sqrt()
        lam1 = u - c
        lam3 = u + c
        mach = u / c.clamp(min=eps)
        old_feats = torch.stack([rho, u, p, c, lam1, lam3, mach], dim=-1)

        assert torch.allclose(new_feats, old_feats, atol=1e-5), (
            f"Max diff: {(new_feats - old_feats).abs().max().item()}"
        )


class TestEulerBiasEquivalence:
    """PDEBias(EulerPDE) matches EulerBias output."""

    @pytest.mark.parametrize("gamma", [1.4, 1.2])
    def test_bias_matches(self, gamma):
        old_mod = EulerBias(gamma=gamma)
        pde = EulerPDE(gamma=gamma)
        new_mod = PDEBias(pde=pde, use_damping=False)

        ic = _make_euler_ic()
        query = _make_query()

        with torch.no_grad():
            old_out = old_mod(ic, query)
            new_out = new_mod(ic, query)

        assert old_out.shape == new_out.shape
        assert torch.allclose(old_out, new_out, atol=1e-4), (
            f"Max diff: {(old_out - new_out).abs().max().item()}"
        )

    def test_bias_with_masking(self):
        ic = _make_euler_ic(B=1, K=3)
        ic["pieces_mask"] = torch.tensor([[1.0, 1.0, 0.0]])

        old_mod = EulerBias(gamma=1.4)
        pde = EulerPDE(gamma=1.4)
        new_mod = PDEBias(pde=pde, use_damping=False)

        query = _make_query(B=1)

        with torch.no_grad():
            old_out = old_mod(ic, query)
            new_out = new_mod(ic, query)

        assert torch.allclose(old_out, new_out, atol=1e-4)


# ===========================================================================
# Collision times (LWR only)
# ===========================================================================


class TestLWRCollisionTimes:
    """LWRPDE.collision_times matches LWRBias._compute_collision_times."""

    def test_collision_times_match(self):
        flux = GreenshieldsFlux()
        pde = LWRPDE(flux=flux)
        ic = _make_lwr_ic()

        new_t_coll = pde.collision_times(ic)

        # Old: static method
        xs = ic["xs"]
        ks = ic["ks"]
        lam = flux.derivative(ks)
        K = ks.shape[1]
        old_t_coll = LWRBias._compute_collision_times(xs, ks, lam, K)

        assert torch.allclose(new_t_coll, old_t_coll, atol=1e-6)

    def test_arz_returns_none(self):
        pde = ARZPDE()
        ic = _make_arz_ic()
        assert pde.collision_times(ic) is None

    def test_euler_returns_none(self):
        pde = EulerPDE()
        ic = _make_euler_ic()
        assert pde.collision_times(ic) is None
