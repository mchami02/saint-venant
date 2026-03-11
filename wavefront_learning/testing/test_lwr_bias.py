"""Standalone tests for LWRBias module.

Uses known IC configurations (no data pipeline dependency).
Run: cd wavefront_learning && uv run python -m pytest testing/test_lwr_bias.py -v
"""

import torch
import pytest

from models.base.lwr_bias import LWRBias
from models.base.flux import GreenshieldsFlux


@pytest.fixture
def flux():
    return GreenshieldsFlux()


def _make_ic(xs, ks):
    """Build ic_data dict from plain lists (single-sample batch)."""
    xs_t = torch.tensor([xs], dtype=torch.float32)
    ks_t = torch.tensor([ks], dtype=torch.float32)
    pieces_mask = torch.ones(1, len(ks), dtype=torch.float32)
    return {"xs": xs_t, "ks": ks_t, "pieces_mask": pieces_mask}


def _query_at(t_vals, x_vals):
    """Build (t_coords, x_coords) each of shape (1, nt, nx)."""
    t_vals = torch.tensor(t_vals, dtype=torch.float32)
    x_vals = torch.tensor(x_vals, dtype=torch.float32)
    # (1, nt, nx) with nt=1
    t_coords = t_vals.unsqueeze(0).unsqueeze(0)  # (1, 1, nx)
    x_coords = x_vals.unsqueeze(0).unsqueeze(0)  # (1, 1, nx)
    return t_coords, x_coords


class TestShockBiasAtT0:
    """2-segment shock at x=0.5: rho_L=0.2 > rho_R=0.8 (lam_L > lam_R)."""

    def test_query_left_of_boundary(self, flux):
        """Query at x=0.2 is outside left segment's boundary -> penalized."""
        bias_mod = LWRBias(flux=flux, use_damping=False)
        # rho_L=0.2, rho_R=0.8 -> lam_L=0.6, lam_R=-0.6 -> shock
        ic = _make_ic([0.0, 0.5, 1.0], [0.2, 0.8])
        t, x = _query_at([0.0], [0.2])
        bias = bias_mod(ic, (t, x))  # (1, 1, 1, 2)
        # Left segment (k=0): boundary at x=0.5 at t=0, query at 0.2 < 0.5
        # dist = x - boundary = 0.2 - 0.5 = -0.3, relu(-0.3) = 0 -> bias = 0
        assert bias[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_query_at_boundary(self, flux):
        """Query at x=0.5 (exactly at boundary) -> zero bias for left seg."""
        bias_mod = LWRBias(flux=flux, use_damping=False)
        ic = _make_ic([0.0, 0.5, 1.0], [0.2, 0.8])
        t, x = _query_at([0.0], [0.5])
        bias = bias_mod(ic, (t, x))
        assert bias[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_query_right_of_boundary(self, flux):
        """Query at x=0.8 is past left segment's boundary -> penalized."""
        bias_mod = LWRBias(flux=flux, use_damping=False)
        ic = _make_ic([0.0, 0.5, 1.0], [0.2, 0.8])
        t, x = _query_at([0.0], [0.8])
        bias = bias_mod(ic, (t, x))
        # Left segment (k=0): dist = 0.8 - 0.5 = 0.3, bias = -0.3
        assert bias[0, 0, 0, 0].item() == pytest.approx(-0.3, abs=1e-5)
        # Right segment (k=1): query inside -> should be 0
        assert bias[0, 0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)


class TestRarefactionFanUnpenalized:
    """Rarefaction: rho_L=0.8, rho_R=0.2 -> lam_L=-0.6 < lam_R=0.6."""

    def test_fan_interior_at_t(self, flux):
        """Query inside the fan region should get 0 bias for both segments."""
        bias_mod = LWRBias(flux=flux, use_damping=False)
        ic = _make_ic([0.0, 0.5, 1.0], [0.8, 0.2])
        # lam_L = 1 - 2*0.8 = -0.6, lam_R = 1 - 2*0.2 = 0.6
        # At t=0.1: left boundary = 0.5 + lam_R*0.1 = 0.56 (far fan edge for left seg)
        #           right boundary = 0.5 + lam_L*0.1 = 0.44 (far fan edge for right seg)
        # Fan spans [0.44, 0.56]. Query at 0.5 is inside the fan.
        t, x = _query_at([0.1], [0.5])
        bias = bias_mod(ic, (t, x))
        # Left seg: dist = 0.5 - 0.56 = -0.06 -> relu = 0
        assert bias[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        # Right seg: dist = 0.44 - 0.5 = -0.06 -> relu = 0
        assert bias[0, 0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)


class TestBiasAtLaterTime:
    """Boundary moves with shock speed; distances shift accordingly."""

    def test_shock_boundary_moves(self, flux):
        """Shock at x=0.5 with speed s. At t>0, boundary has shifted."""
        bias_mod = LWRBias(flux=flux, use_damping=False)
        ic = _make_ic([0.0, 0.5, 1.0], [0.2, 0.8])
        # shock speed s = 1 - 0.2 - 0.8 = 0.0 (Greenshields analytical)
        # At t=0.5: boundary = 0.5 + 0.0*0.5 = 0.5
        t, x = _query_at([0.5], [0.7])
        bias = bias_mod(ic, (t, x))
        # Left seg: dist = 0.7 - 0.5 = 0.2
        assert bias[0, 0, 0, 0].item() == pytest.approx(-0.2, abs=1e-5)

    def test_nonzero_shock_speed(self, flux):
        """Shock with nonzero speed shifts the boundary over time."""
        bias_mod = LWRBias(flux=flux, use_damping=False)
        # rho_L=0.1, rho_R=0.6 -> lam_L=0.8, lam_R=-0.2 -> shock
        # s = 1 - 0.1 - 0.6 = 0.3
        ic = _make_ic([0.0, 0.5, 1.0], [0.1, 0.6])
        # At t=1.0: boundary = 0.5 + 0.3*1.0 = 0.8
        t, x = _query_at([1.0], [0.9])
        bias = bias_mod(ic, (t, x))
        # Left seg: dist = 0.9 - 0.8 = 0.1
        assert bias[0, 0, 0, 0].item() == pytest.approx(-0.1, abs=1e-4)


class TestDampingOnVsOff:
    """With damping: bias fades after collision time. Without: constant."""

    def test_damping_off_constant(self, flux):
        """Without damping, bias does not fade even at large t."""
        bias_mod = LWRBias(flux=flux, use_damping=False)
        ic = _make_ic([0.0, 0.5, 1.0], [0.2, 0.8])
        # Query far in time at same spatial point
        t_early, x = _query_at([0.0], [0.8])
        t_late, _ = _query_at([10.0], [0.8])
        bias_early = bias_mod(ic, (t_early, x))
        bias_late = bias_mod(ic, (t_late, x))
        # Without damping, the distance-based part is the same
        # (shock speed = 0 for this IC, so boundary stays at 0.5)
        assert bias_early[0, 0, 0, 0].item() == pytest.approx(
            bias_late[0, 0, 0, 0].item(), abs=1e-5
        )

    def test_damping_on_fades(self, flux):
        """With damping, bias magnitude decreases after collision time."""
        bias_mod = LWRBias(
            flux=flux, use_damping=True, initial_damping_sharpness=10.0
        )
        ic = _make_ic([0.0, 0.5, 1.0], [0.2, 0.8])
        t_early, x = _query_at([0.01], [0.8])
        t_late, _ = _query_at([100.0], [0.8])
        bias_early = bias_mod(ic, (t_early, x))
        bias_late = bias_mod(ic, (t_late, x))
        # Early bias should be more negative than late (late is damped toward 0)
        assert abs(bias_early[0, 0, 0, 0].item()) > abs(
            bias_late[0, 0, 0, 0].item()
        )


class TestMasking:
    """Padded segments (pieces_mask=0) get -1e9."""

    def test_padded_segment_bias(self, flux):
        bias_mod = LWRBias(flux=flux, use_damping=False)
        xs = torch.tensor([[0.0, 0.5, 1.0, 1.5]], dtype=torch.float32)
        ks = torch.tensor([[0.2, 0.8, 0.0]], dtype=torch.float32)
        pieces_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
        ic = {"xs": xs, "ks": ks, "pieces_mask": pieces_mask}
        t, x = _query_at([0.0], [0.5])
        bias = bias_mod(ic, (t, x))
        # Third segment is masked -> should be -1e9
        assert bias[0, 0, 0, 2].item() == pytest.approx(-1e9, rel=1e-3)


class TestOutputShape:
    """Output is (B, nt, nx, K) for various sizes."""

    @pytest.mark.parametrize(
        "B,nt,nx,K",
        [(1, 1, 1, 2), (2, 5, 10, 3), (1, 3, 7, 4)],
    )
    def test_shape(self, flux, B, nt, nx, K):
        bias_mod = LWRBias(flux=flux, use_damping=True)
        xs = torch.rand(B, K + 1).cumsum(dim=1)
        ks = torch.rand(B, K)
        pieces_mask = torch.ones(B, K)
        ic = {"xs": xs, "ks": ks, "pieces_mask": pieces_mask}
        t_coords = torch.rand(B, nt, nx)
        x_coords = torch.rand(B, nt, nx)
        bias = bias_mod(ic, (t_coords, x_coords))
        assert bias.shape == (B, nt, nx, K)
