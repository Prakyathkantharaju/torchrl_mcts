from typing import cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from mcts.mcts_policy import (
    UpdateTreeStrategy,
    AlphaZeroExpansionStrategy,
    MctsPolicyWithTreeUpdate
)
from mcts.stateless_env import StatelessGymEnv
from mcts.sim_transform import SimTransform

from mcts_policy import MctsPolicyWithTreeUpdate, AlphaZeroExpansionStrategy, UpdateTreeStrategy
from tensordict_map import TensorDictMap
from tensordict.nn import TensorDictModule


from torchrl.envs.transforms import TransformedEnv
from stateless_env import StatelessGymEnv

def test_stateless_env_wrapper():
    stateless_env = StatelessGymEnv('Taxi-v3', get_state = None, set_state = None)
    stateless_env.reset()
    out_tensordict = stateless_env.rollout(1)
    assert 'state' in out_tensordict.keys()



def test_stateless_force_state_update():
    stateless_env = StatelessGymEnv('Taxi-v3', get_state = None, set_state = None)
    stateless_env.reset()
    state = 0
    input_tensordict = TensorDict({'state': torch.tensor(state)}, batch_size=())
    output_tensordict = stateless_env.force_state_update(input_tensordict)
    assert output_tensordict['state'] == state

def test_sim_transform():
    stateless_env = StatelessGymEnv('Taxi-v3', get_state = None, set_state = None)
    # This is with random policy.
    env = TransformedEnv(stateless_env, SimTransform(base_env = stateless_env, number_of_simulations = 1))
    out_tensordict = env.rollout(1) 
    assert 'rollout' in out_tensordict.keys()



def test_sim_transform():
    stateless_env = StatelessGymEnv('Taxi-v3', get_state = None, set_state = None)
    # This is with random policy.
    env = TransformedEnv(stateless_env, SimTransform(base_env = stateless_env, number_of_simulations = 1))
    out_tensordict = env.rollout(1) 
    assert 'rollout' in out_tensordict.keys()


def test_policy_sim_transform():
    stateless_env = StatelessGymEnv('Taxi-v3', get_state = None, set_state = None)
    env = TransformedEnv(stateless_env, SimTransform(base_env = stateless_env, number_of_simulations = 1))
    tree = TensorDictMap("observation")
    value_module = TensorDictModule(
        module=lambda x: torch.zeros(env.action_spec.shape),
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    expansion_strategy = AlphaZeroExpansionStrategy(
        tree=tree,
        value_module=value_module,
    )
    tree_updater = UpdateTreeStrategy(tree)
    policy = MctsPolicyWithTreeUpdate(
        expansion_strategy=expansion_strategy,
        tree_update_strategy=tree_updater,
    )
    output = env.rollout(1, policy=policy)
    assert 'rollout' in output.keys()


