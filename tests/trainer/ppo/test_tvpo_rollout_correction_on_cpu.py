# Copyright 2026 Individual Contributor: Michael Noukhovitch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TVPO gradients and prompt metadata through the shared engine loss interface."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_tvpo, wrap_rollout_correction
from verl.trainer.ppo.rollout_corr_helper import (
    apply_bypass_mode,
    compute_rollout_correction_and_rejection_mask,
)
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, reduce_metrics
from verl.workers.config import ActorConfig, PolicyLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding


def _config(loss_mode="tvpo", **correction):
    return ActorConfig(
        strategy="fsdp",
        rollout_n=2,
        ppo_micro_batch_size_per_gpu=2,
        clip_ratio=0.8,
        clip_ratio_c=2.0,
        policy_loss=PolicyLossConfig(loss_mode=loss_mode, rollout_correction=OmegaConf.create(correction)),
    )


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("loss_mode", ["tvpo", "vanilla"])
def test_engine_binds_prompt_ids_only_for_tvpo(bypass, loss_mode):
    # One response has TV=.6 and one has TV=0: the shared prompt is below .4.
    # Dropping uid gates the first response and therefore changes its gradient.
    ratios = torch.tensor([[2.2, 2.2], [1.0, 1.0]])
    leaf = ratios.log().requires_grad_()
    old = torch.full_like(leaf, -3.0)
    config = _config("bypass_mode" if bypass else loss_mode, loss_type="tvpo" if loss_mode == "tvpo" else "ppo_clip")
    data = TensorDict(
        {
            "prompts": torch.zeros(2, 1, dtype=torch.long),
            "responses": torch.zeros(2, 2, dtype=torch.long),
            "input_ids": torch.zeros(2, 3, dtype=torch.long),
            "position_ids": torch.arange(3).expand(2, -1),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "response_mask": torch.ones(2, 2),
            "old_log_probs": old,
            "advantages": torch.ones(2, 2),
        },
        batch_size=[2],
    )
    data["uid"] = tu.get_tensordict({"uid": ["prompt-a", "prompt-a"]})["uid"]
    if not bypass:
        data["rollout_is_weights"] = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=4, global_batch_size=2)
    data = left_right_2_no_padding(data)
    # Engine outputs are shifted by one prediction token when unpacked.
    packed = torch.cat([leaf + old, torch.zeros(2, 1)], dim=1)
    packed = torch.nested.as_nested_tensor(list(packed.unbind()), layout=torch.jagged)
    loss, metrics = ppo_loss(config, {"log_probs": packed}, data)
    loss.backward()
    weights = 1.0 if bypass else torch.tensor([[0.5, 0.5], [1.5, 1.5]])
    if loss_mode == "tvpo":
        torch.testing.assert_close(leaf.grad, -ratios.clamp(max=2) * weights / 4)
        assert metrics["actor/ppo_tv"].aggregate() == pytest.approx(0.3)
        assert metrics["actor/ppo_tv_prompt/hist"] == pytest.approx([0.3])
        assert metrics["actor/ppo_tv_prompt/min"].aggregation == AggregationType.MIN
        assert metrics["actor/ppo_tv_prompt/max"].aggregation == AggregationType.MAX
    else:
        # Ordinary PPO accepts no prompt metadata and retains its original clipping.
        torch.testing.assert_close(leaf.grad, -torch.tensor([[0.0, 0.0], [1.0, 1.0]]) * weights / 4)
        assert not any("tv" in key for key in metrics)


@pytest.mark.parametrize("rollout_is", [None, "token", "sequence"])
def test_bypass_tvpo_applies_rejection_without_double_weighting(rollout_is):
    old = torch.full((2, 3), -3.0)
    current = (old + torch.tensor([[1.1, -0.2, 0.0], [0.1, -0.1, 0.0]])).requires_grad_()
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    advantages = torch.tensor([[1.0, -1.0, 9.0], [1.0, -1.0, 9.0]])
    config = _config(
        "tvpo",
        bypass_mode=True,
        loss_type="tvpo",
        rollout_is=rollout_is,
        rollout_is_threshold=1.5,
        rollout_rs="token_k1",
        rollout_rs_threshold="0.5_2.0",
    )
    batch = DataProto.from_dict({"rollout_log_probs": old})
    policy_config = OmegaConf.create({"loss_mode": "tvpo"})
    apply_bypass_mode(batch, config.policy_loss.rollout_correction, policy_config)
    assert policy_config.loss_mode == "bypass_mode"
    torch.testing.assert_close(batch.batch["old_log_probs"], old)
    config = replace(config, policy_loss=policy_config)
    _, effective_mask, _ = compute_rollout_correction_and_rejection_mask(
        old_log_prob=current.detach(),
        rollout_log_prob=old,
        response_mask=mask,
        rollout_is=rollout_is,
        rollout_is_threshold=1.5,
        rollout_rs="token_k1",
        rollout_rs_threshold="0.5_2.0",
    )
    assert effective_mask[0, 0] == 0
    index = np.array(["a", "a"])
    loss_fn = wrap_rollout_correction(partial(compute_policy_loss_tvpo, index=index))
    loss, metrics = loss_fn(
        old,
        current,
        advantages,
        mask,
        config=config,
        rollout_is_weights=torch.full_like(mask, 42.0),
    )
    expected, _ = compute_policy_loss_tvpo(
        old,
        current,
        advantages,
        effective_mask,
        config=config,
        index=index,
    )
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, current, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, current)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[0, 0] == 0
    assert torch.count_nonzero(actual_grad[:, 2]) == 0
    assert any(key.startswith("rollout_corr/") for key in metrics)


def test_tvpo_gates_only_gradients_that_increase_tv_and_handles_rejected_groups():
    old = torch.full((3, 2), -3.0)
    ratios = torch.tensor([[3.0, 0.5], [3.0, 0.5], [100.0, 100.0]])
    current = (old + ratios.log()).requires_grad_()
    advantages = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    loss, metrics = compute_policy_loss_tvpo(
        old,
        current,
        advantages,
        mask,
        config=_config(),
        index=torch.tensor([100, 100, 999]),
    )
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[0.0, 0.0], [0.5, -0.125], [0.0, 0.0]]))
    assert metrics["actor/ppo_tv_prompt/hist"] == pytest.approx([0.625])
    empty_loss, empty_metrics = compute_policy_loss_tvpo(
        old,
        current,
        advantages,
        torch.zeros_like(mask),
        config=_config(),
        index=torch.tensor([100, 100, 999]),
    )
    assert empty_loss.item() == 0
    assert empty_metrics["actor/ppo_tv_prompt/hist"] == [0.0]


def test_tv_histogram_survives_uneven_microbatches_and_ranks():
    metrics = {"actor/ppo_tv_prompt/hist": [[[0.1], [0.2, 0.3]], [[0.4, 0.5, 0.6]]]}
    assert reduce_metrics(metrics)["actor/ppo_tv_prompt/hist"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
