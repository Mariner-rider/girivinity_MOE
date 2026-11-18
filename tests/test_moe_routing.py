"""Comprehensive test suite for Section 5.4: Mixture-of-Experts (MoE) Routing.

Verifies:
1. Router shapes: gate weight, bias, logits, top-k probabilities, indices, and output shapes.
2. Top-k selection: distinct expert selection, logit alignment, and weight normalization (sum to 1.0).
3. Expert dispatch and token ordering: batched sorted execution vs un-batched reference loop.
4. Shared experts: unconditional execution (weight=1.0), additive contribution, and zero-shared bypass.
5. Capacity and overflow policy: dropless dynamic capacity vs bounded capacity factor with overflow drops.
6. Load balancing: auxiliary-loss-free bias adaptation, zero-centering, train vs eval freeze.
7. Gradient isolation: active experts receive gradients, inactive experts have grad=None.
8. Parameter count vs active parameters: inactive experts count toward total parameters but NOT active-per-token count.
9. Determinism: bitwise identical routing decisions and gradients across multiple runs.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from model.architecture import Expert, GirivinityConfig, GirivinityModel, MoELayer


def _make_moe_cfg(**overrides) -> GirivinityConfig:
    defaults = dict(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=200,
        max_seq_len=32,
        ffn_multiplier=2.0,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        moe_intermediate_size=32,
        moe_start_layer=1,
    )
    defaults.update(overrides)
    return GirivinityConfig(**defaults)


# =====================================================================
# 1. Router Shapes
# =====================================================================
def test_moe_router_shapes():
    """Validates tensor shapes across all intermediate router projections and final output."""
    cfg = _make_moe_cfg(n_routed_experts=8, n_activated_experts=2, dim=64)
    moe = MoELayer(cfg)
    moe.eval()

    # Gate weight shape
    assert moe.gate.weight.shape == (cfg.n_routed_experts, cfg.dim)
    # Routing bias shape
    assert moe.routing_bias.shape == (cfg.n_routed_experts,)

    B, T, D = 3, 7, cfg.dim
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = moe(x)

    # Output matches input shape exactly
    assert out.shape == (B, T, D)

    # Stashed router logits shape: [B*T, n_routed]
    assert hasattr(moe, "_last_router_logits")
    assert moe._last_router_logits.shape == (B * T, cfg.n_routed_experts)


# =====================================================================
# 2. Top-k Selection and Normalization
# =====================================================================
def test_moe_topk_selection_and_weight_normalization():
    """Verifies that top-k selection picks the highest logits without duplicates and normalizes weights."""
    cfg = _make_moe_cfg(n_routed_experts=8, n_activated_experts=3, dim=64)
    moe = MoELayer(cfg)
    moe.eval()

    N = 10
    x_flat = torch.randn(N, cfg.dim)
    with torch.no_grad():
        logits = moe.gate(x_flat) + moe.routing_bias
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, moe.top_k, dim=-1)
        normalized_w = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    # Exactly top_k indices per token
    assert topk_idx.shape == (N, cfg.n_activated_experts)
    assert normalized_w.shape == (N, cfg.n_activated_experts)

    # Indices within valid expert range
    assert (topk_idx >= 0).all() and (topk_idx < cfg.n_routed_experts).all()

    # Check each token individually
    for n in range(N):
        indices = topk_idx[n].tolist()
        # No duplicate experts for the same token
        assert len(set(indices)) == moe.top_k, f"Token {n} selected duplicate experts: {indices}"

        # Selected indices must match the indices of the largest logits
        expected_topk = torch.topk(logits[n], moe.top_k).indices.tolist()
        assert indices == expected_topk

        # Routing weights must sum to 1.0 (convex combination)
        weight_sum = normalized_w[n].sum().item()
        assert abs(weight_sum - 1.0) < 1e-6, f"Weights sum to {weight_sum}, expected 1.0"

        # All weights strictly positive
        assert (normalized_w[n] > 0).all()


# =====================================================================
# 3. Expert Dispatch and Token Ordering
# =====================================================================
@pytest.mark.parametrize("batch_size,seq_len", [(1, 8), (3, 5), (4, 12)])
def test_moe_expert_dispatch_matches_naive_token_by_token_reference(batch_size: int, seq_len: int):
    """Proves that grouped/sorted expert dispatch preserves exact token ordering and weights.

    Compares the fast sorted-boundaries index_add dispatch against an explicit un-batched
    token-by-token reference loop.
    """
    torch.manual_seed(42 + batch_size)
    cfg = _make_moe_cfg(n_routed_experts=6, n_activated_experts=2, n_shared_experts=1, dim=64)
    moe = MoELayer(cfg)
    moe.eval()

    x = torch.randn(batch_size, seq_len, cfg.dim)
    with torch.no_grad():
        out_batched = moe(x)

        # Naive un-batched reference loop
        x_flat = x.reshape(-1, cfg.dim)
        logits = moe.gate(x_flat) + moe.routing_bias
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, moe.top_k, dim=-1)
        normalized_w = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out_ref = torch.zeros_like(x_flat)
        for n in range(x_flat.size(0)):
            # Sum routed expert contributions
            for k in range(moe.top_k):
                e = topk_idx[n, k].item()
                w = normalized_w[n, k]
                out_ref[n] += w * moe.routed_experts[e](x_flat[n : n + 1]).squeeze(0)
            # Sum shared expert contributions
            if moe.shared_experts is not None:
                for shared in moe.shared_experts:
                    out_ref[n] += shared(x_flat[n : n + 1]).squeeze(0)

        out_ref = out_ref.reshape(batch_size, seq_len, cfg.dim)

    # Token ordering and dispatch equivalence verified within float32 machine precision
    max_diff = (out_batched - out_ref).abs().max().item()
    assert (
        max_diff < 1e-6
    ), f"Dispatch mismatch: max diff {max_diff} between batched dispatch and token reference"


# =====================================================================
# 4. Shared Experts
# =====================================================================
def test_moe_shared_experts_additive_contribution_and_zero_shared_bypass():
    """Verifies that shared experts contribute additively with weight 1.0 and zero-shared is supported."""
    torch.manual_seed(100)
    # Config with 2 shared experts
    cfg_with_shared = _make_moe_cfg(n_routed_experts=4, n_activated_experts=2, n_shared_experts=2, dim=64)
    moe = MoELayer(cfg_with_shared)
    moe.eval()

    x = torch.randn(2, 4, 64)
    with torch.no_grad():
        out = moe(x)
        # Compute routed part alone (temporarily disable shared experts)
        shared_save = moe.shared_experts
        moe.shared_experts = None
        out_routed_only = moe(x)
        moe.shared_experts = shared_save

        # Compute shared parts alone
        shared_contrib = torch.zeros_like(x)
        x_flat = x.reshape(-1, 64)
        for shared in moe.shared_experts:
            shared_contrib += shared(x_flat).reshape(2, 4, 64)

    # Output with shared experts must exactly equal routed + shared
    assert torch.allclose(out, out_routed_only + shared_contrib, atol=1e-6)

    # Config with zero shared experts
    cfg_no_shared = _make_moe_cfg(n_routed_experts=4, n_activated_experts=2, n_shared_experts=0, dim=64)
    moe_no_shared = MoELayer(cfg_no_shared)
    assert moe_no_shared.shared_experts is None
    with torch.no_grad():
        out_no_shared = moe_no_shared(x)
    assert out_no_shared.shape == (2, 4, 64)


# =====================================================================
# 5. Capacity and Overflow Policy
# =====================================================================
def test_moe_dropless_capacity_policy_default():
    """Confirms default policy is dropless (dynamic capacity): zero tokens dropped."""
    cfg = _make_moe_cfg(n_routed_experts=8, n_activated_experts=2)
    assert cfg.moe_capacity_factor is None
    moe = MoELayer(cfg)
    moe.eval()

    x = torch.randn(4, 16, 64)
    with torch.no_grad():
        moe(x)
    assert moe.last_overflow_count == 0


def test_moe_bounded_capacity_and_overflow_drop_policy():
    """Verifies that setting a capacity factor enforces an expert ceiling and drops overflow tokens."""
    torch.manual_seed(200)
    # 8 experts, top-2. With 32 tokens, total assignments = 64. Average per expert = 8 tokens.
    # Set capacity_factor = 0.5 -> max capacity per expert = ceil(8 * 0.5) = 4 tokens.
    cfg = _make_moe_cfg(
        n_routed_experts=8,
        n_activated_experts=2,
        moe_capacity_factor=0.5,
    )
    moe = MoELayer(cfg)
    moe.eval()

    # Skew router weights strongly toward expert 0 so expert 0 receives > 4 tokens
    with torch.no_grad():
        moe.gate.weight[0].fill_(10.0)

    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        out = moe(x)

    assert out.shape == (2, 16, 64)
    # Overflow tokens must be detected and dropped
    assert moe.last_overflow_count > 0, "Expected overflow tokens when skewed expert exceeds capacity ceiling"


# =====================================================================
# 6. Load Balancing and Bias Movement
# =====================================================================
def test_moe_load_balancing_bias_movement_and_eval_freeze():
    """Verifies that observed load updates routing bias during training, stays zero-centered, and freezes in eval."""
    torch.manual_seed(300)
    cfg = _make_moe_cfg(n_routed_experts=8, n_activated_experts=2, moe_bias_update_speed=0.01)
    moe = MoELayer(cfg)

    # Initial bias is all zeros
    assert torch.equal(moe.routing_bias, torch.zeros(8))

    # In eval mode: bias should NOT update
    moe.eval()
    x = torch.randn(4, 8, 64)
    with torch.no_grad():
        moe(x)
    assert torch.equal(moe.routing_bias, torch.zeros(8)), "Bias updated while in eval() mode!"

    # In train mode: bias should update based on load
    moe.train()
    moe(x)
    assert not torch.equal(moe.routing_bias, torch.zeros(8)), "Bias failed to update during training!"

    # Bias remains zero-centered
    bias_sum = moe.routing_bias.sum().item()
    assert abs(bias_sum) <= moe.bias_update_speed * cfg.n_routed_experts, f"Bias sum {bias_sum} deviated from zero-center"

    # Heavily loaded expert has its bias decreased
    # Run another forward with extreme skew toward expert 0
    with torch.no_grad():
        moe.gate.weight[0].fill_(5.0)
    for _ in range(5):
        moe(x)
    # Expert 0 was consistently loaded above average, so its bias must be negative
    assert moe.routing_bias[0].item() < 0.0, "Overloaded expert bias should decrease below 0"


# =====================================================================
# 7. Gradient Isolation Across Experts
# =====================================================================
def test_moe_expert_gradient_isolation():
    """Validates that inactive experts receive ZERO gradients, while active experts and gate receive gradients."""
    torch.manual_seed(400)
    cfg = _make_moe_cfg(n_routed_experts=8, n_activated_experts=2, n_shared_experts=1, dim=64)
    moe = MoELayer(cfg)
    moe.train()

    # Pass a single token: exactly 2 experts are active, 6 experts are completely inactive
    x = torch.randn(1, 1, 64, requires_grad=True)
    out = moe(x)
    out.sum().backward()

    # Find which 2 experts were chosen
    logits = moe._last_router_logits
    probs = F.softmax(logits, dim=-1)
    _, topk_idx = torch.topk(probs, moe.top_k, dim=-1)
    active_experts = set(topk_idx.squeeze().tolist())
    assert len(active_experts) == 2

    # Verify gradients for all routed experts
    for e in range(cfg.n_routed_experts):
        grad = moe.routed_experts[e].gate.weight.grad
        if e in active_experts:
            assert grad is not None, f"Active expert {e} received no gradient!"
            assert (grad != 0).any(), f"Active expert {e} gradient was zero!"
        else:
            assert grad is None, f"Inactive expert {e} received gradient ({grad})! Gradient leakage detected."

    # Shared experts process all tokens -> must have gradient
    for s_idx, shared in enumerate(moe.shared_experts):
        assert shared.gate.weight.grad is not None
        assert (shared.gate.weight.grad != 0).any()

    # Router gate must receive gradient through the routing probabilities path
    assert moe.gate.weight.grad is not None
    assert (moe.gate.weight.grad != 0).any()


# =====================================================================
# 8. Inactive Experts Count Toward Total Parameters But NOT Active
# =====================================================================
def test_moe_parameter_counts_total_vs_active():
    """Confirms inactive experts count toward total parameters but not active-per-token count."""
    dim = 64
    intermediate = 32
    n_routed = 8
    top_k = 2
    n_shared = 1

    cfg = _make_moe_cfg(
        dim=dim,
        moe_intermediate_size=intermediate,
        n_routed_experts=n_routed,
        n_activated_experts=top_k,
        n_shared_experts=n_shared,
        n_layers=4,
        moe_start_layer=1,
    )
    model = GirivinityModel(cfg)

    # 1. MoE Layer parameter accounting
    moe_layer = model.layers[1].ffn
    assert isinstance(moe_layer, MoELayer)

    expert_p = 3 * dim * intermediate  # gate, up, down
    gate_p = dim * n_routed
    shared_p = n_shared * expert_p

    expected_total_moe = gate_p + shared_p + (n_routed * expert_p)
    expected_active_moe = gate_p + shared_p + (top_k * expert_p)
    expected_inactive_moe = (n_routed - top_k) * expert_p

    assert moe_layer.count_total_parameters() == expected_total_moe
    assert moe_layer.count_active_parameters() == expected_active_moe
    assert moe_layer.count_inactive_parameters() == expected_inactive_moe
    assert (
        moe_layer.count_total_parameters()
        == moe_layer.count_active_parameters() + moe_layer.count_inactive_parameters()
    )

    # 2. Model-wide parameter accounting
    total_model_p = model.count_parameters()
    active_model_p = model.count_active_parameters()
    inactive_model_p = model.count_inactive_parameters()

    assert total_model_p > active_model_p
    # Exactly 3 MoE layers (layers 1, 2, 3):
    num_moe_layers = 3
    expected_total_inactive = num_moe_layers * expected_inactive_moe
    assert inactive_model_p == expected_total_inactive
    assert total_model_p == active_model_p + inactive_model_p


# =====================================================================
# 9. Deterministic Routing and Gradients
# =====================================================================
def test_moe_deterministic_routing_and_gradients():
    """Tests that identical inputs and weights produce bitwise deterministic routing and gradients."""
    torch.manual_seed(500)
    cfg = _make_moe_cfg(n_routed_experts=8, n_activated_experts=2, n_shared_experts=1, dim=64)

    # Initialize two models with identical weights
    moe1 = MoELayer(cfg)
    moe1.train()

    moe2 = MoELayer(cfg)
    moe2.load_state_dict(moe1.state_dict())
    moe2.train()

    x1 = torch.randn(2, 6, cfg.dim, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)

    # Forward pass
    out1 = moe1(x1)
    out2 = moe2(x2)

    # Output tensors must be bitwise identical
    assert torch.equal(out1, out2), "Forward outputs were not bitwise identical!"
    # Router logits must be bitwise identical
    assert torch.equal(moe1._last_router_logits, moe2._last_router_logits)

    # Backward pass
    loss1 = (out1 ** 2).sum()
    loss2 = (out2 ** 2).sum()
    loss1.backward()
    loss2.backward()

    # Gradients must be bitwise identical across all components
    assert torch.equal(x1.grad, x2.grad), "Input gradients differed!"
    assert torch.equal(moe1.gate.weight.grad, moe2.gate.weight.grad), "Router gate gradients differed!"

    for e in range(cfg.n_routed_experts):
        g1 = moe1.routed_experts[e].gate.weight.grad
        g2 = moe2.routed_experts[e].gate.weight.grad
        if g1 is None:
            assert g2 is None
        else:
            assert torch.equal(g1, g2), f"Expert {e} gradients differed!"

    for s in range(cfg.n_shared_experts):
        sg1 = moe1.shared_experts[s].gate.weight.grad
        sg2 = moe2.shared_experts[s].gate.weight.grad
        assert torch.equal(sg1, sg2), f"Shared expert {s} gradients differed!"
