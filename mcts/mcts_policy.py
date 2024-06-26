from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Iterable
from copy import copy

import torch
from tensordict import TensorDictBase, TensorDict, NestedKey
from tensordict.nn import TensorDictModule, TensorDictSequential

# noinspection PyProtectedMember
from tensordict.nn.common import TensorDictModuleBase
from torchrl.envs import EnvBase
from torchrl.envs.utils import exploration_type, ExplorationType, set_exploration_type
from torchrl.objectives.value import ValueEstimatorBase, TDLambdaEstimator

from tensordict_map import TensorDictMap

from tqdm import tqdm

# temp import need to be removed after debug
import numpy as np
from collections import defaultdict

def safe_weighted_avg(
    w1: torch.Tensor,
    v1: torch.Tensor,
    w2: torch.Tensor,
    v2: torch.Tensor,
):
    total_weight = w1 + w2
    mask = total_weight > 0
    total_sum = w1 * v1 + w2 * v2
    weighted_avg = torch.zeros_like(v1)
    weighted_avg[mask] = total_sum[mask] / total_weight[mask]
    return weighted_avg


class MaxActionValue(TensorDictModuleBase):
    def __init__(
        self,
        tree: TensorDictMap,
        action_value_key: str = "action_value",
        state_value_key: str = "state_value",
    ):
        self.in_keys = tree.keys
        self.out_keys = [state_value_key]
        super().__init__()
        self.tree = tree
        self.state_value_key = state_value_key
        self.action_value_key = action_value_key

    def forward(self, tensordict: TensorDict):
        node = self.tree.get(tensordict)

        if node is None:
            state_value = torch.zeros(tensordict.batch_size + (1,))
        else:
            state_value = torch.argmax(
                node[self.action_value_key], dim=-1, keepdim=True
            )[0]
            if state_value.numel():
                state_value = state_value.view(tensordict.batch_size + (1,))
        tensordict[self.state_value_key] = state_value


def safe_weighted_avg(
    w1: torch.Tensor,
    v1: torch.Tensor,
    w2: torch.Tensor,
    v2: torch.Tensor,
):
    total_weight = w1 + w2
    mask = total_weight > 0
    total_sum = w1 * v1 + w2 * v2
    weighted_avg = torch.max(v1, w2 * v2)
    weighted_avg[mask] = total_sum[mask].to(torch.float32) / total_weight[mask]
    return weighted_avg


class MaxActionValue(TensorDictModuleBase):
    def __init__(
        self,
        tree: TensorDictMap,
        action_value_key: str = "action_value",
        state_value_key: str = "state_value",
    ):
        self.in_keys = tree.keys
        self.out_keys = [state_value_key]
        super().__init__()
        self.tree = tree
        self.state_value_key = state_value_key
        self.action_value_key = action_value_key

    def forward(self, tensordict: TensorDict):
        if len(tensordict.batch_size) != 2:
            raise ValueError(
                f"Input tensordict expected to have a shape of batch x time_step but got {tensordict.batch_size}"
            )
        if tensordict.batch_size[0] != 1:
            raise ValueError(
                f"Current implementation only support batch size of 1 but got {tensordict.batch_size[0]}"
            )
        state_values = []
        for step in tensordict.unbind(dim=1):
            node = self.tree.get(step)

            if node is None:
                state_value = torch.zeros((1,))
            else:
                state_value = torch.max(
                    node[self.action_value_key], dim=-1, keepdim=True
                )[0]
            state_values.append(state_value)

        tensordict[self.state_value_key] = torch.stack(state_values, dim=0).unsqueeze(
            dim=0
        )


class UpdateTreeStrategy:
    """
    The strategy to update tree after each rollout. This class uses the given value estimator
    to compute a target value after each roll out and compute the mean of target values in the tree.

    It also updates the number of time nodes get visited in tree.

    Args:
        tree: A TensorDictMap that store stats of the tree.
        value_estimator: A ValueEstimatorBase that compute target value.
        action_key: A key in the rollout TensorDict to store the selected action.
        action_value_key: A key in the tree nodes that stores the mean of Q(s, a).
        action_count_key: A key in the tree nodes that stores the number of times nodes get visited.
    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        tree: TensorDictMap,
        value_estimator: Optional[ValueEstimatorBase] = None,
        action_key: NestedKey = "action",
        action_value_key: NestedKey = "action_value",
        action_count_key: NestedKey = "action_count",
        chosen_action_value_key: NestedKey = "chosen_action_value",
    ):
        self.tree = tree
        self.action_key = action_key
        self.action_value_key = action_value_key
        self.action_count_key = action_count_key
        self.chosen_action_value = chosen_action_value_key
        self.value_estimator = value_estimator or self.get_default_value_network(tree)

    @staticmethod
    def get_default_value_network(tree: TensorDictMap) -> ValueEstimatorBase:
        # noinspection PyTypeChecker
        return TDLambdaEstimator(
            gamma=1.0,
            lmbda=1.0,
            value_network=MaxActionValue(tree),
            vectorized=False,  # Todo: use True instead and fix the error
        )

    def update(self, rollout: TensorDictBase) -> None:
        tree = self.tree
        action_count_key = self.action_count_key
        action_value_key = self.action_value_key
        r = torch.max(rollout[("next", "reward")]).item()
        # usually time is along the last dimension (if envs are batched for instance)
        steps = rollout.unbind(-1)

        value_estimator_input = rollout.unsqueeze(dim=0)
        target_value = self.value_estimator.value_estimate(value_estimator_input)
        target_value = target_value.squeeze(dim=0)

        target_values = target_value.unbind(rollout.ndim - 1)
        observation = defaultdict(int)
        actions_set = defaultdict(int)
        for idx in range(rollout.batch_size[-1]):
            state = steps[idx]
            node = tree[state]
            action = state[self.action_key]

            node[action_value_key] = safe_weighted_avg(
                node[action_count_key],
                node[action_value_key],
                action,
                target_values[idx],
            )
            node[action_count_key] += action
            tree[state] = node
        #     observation[torch.argmax(node['observation'], dim = -1).detach().numpy().item()] += 1
        #     actions_set[torch.argmax(action, dim = -1).detach().numpy().item()] += 1
        # print(observation, "observation", actions_set, "actions")
        
        # return rollout so that we can train the alpha zero testing.
        return rollout # we are returning this to avoid returning the again during the alphazero coding.

    def start_simulation(self):
        self.tree.clear()


class ExpansionStrategy(TensorDictModuleBase):
    """
    The rollout policy in expanding tree.
    This policy will use to initialize a node when it gets expanded at the first time.
    """

    def __init__(
        self,
        tree: TensorDictMap,
        out_keys: List[str],
        in_keys: Optional[List[str]] = None,
    ):
        self.in_keys = list(set(tree.keys + [] if in_keys is None else in_keys))
        self.out_keys = out_keys
        super().__init__()
        self.tree = tree

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        The node to be expanded. The output Tensordict will be used in future
        to select action.
        Args:
            tensordict: The state that need to be explored

        Returns:
            A initialized statistics to select actions in the future.
        """
        node = self.tree.get(tensordict)
        if node is None:
            node = self.expand(tensordict)
            self.tree[tensordict] = node
            return node

        return node

    @abstractmethod
    def expand(self, tensordict: TensorDictBase) -> TensorDictBase:
        pass


class AlphaZeroExpansionStrategy(ExpansionStrategy):
    """
    An implementation of Alpha Zero to initialize a node at its first time.

    Args:
            value_module: a TensorDictModule to initialize a prior for Q(s, a)
            module_action_value_key: a key in the output of value_module that contains Q(s, a) values
    """

    def __init__(
        self,
        tree: TensorDictMap,
        value_module: TensorDictModule,
        action_value_key: NestedKey = "action_value",
        prior_action_value_key: NestedKey = "prior_action_value",
        action_count_key: NestedKey = "action_count",
        module_action_value_key: NestedKey = "action_value",
    ):
        super().__init__(
            tree=tree,
            out_keys=value_module.out_keys
            + [action_value_key, prior_action_value_key, action_count_key],
            in_keys=value_module.in_keys,
        )
        assert module_action_value_key in value_module.out_keys
        self.value_module = value_module
        self.action_value_key = module_action_value_key
        self.q_sa_key = action_value_key
        self.p_sa_key = prior_action_value_key
        self.n_sa_key = action_count_key

    def expand(self, tensordict: TensorDictBase) -> TensorDict:
        module_output = self.value_module(tensordict)
        p_sa = module_output[self.action_value_key]
        module_output[self.q_sa_key] = torch.clone(p_sa)
        module_output[self.p_sa_key] = p_sa
        module_output[self.n_sa_key] = torch.zeros_like(p_sa)
        return module_output


class PuctSelectionPolicy(TensorDictModuleBase):
    """
    The optimism under uncertainty estimation computed by the PUCT formula in AlphaZero paper:
    https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf

    Args:
        cpuct: A constant to control exploration
        action_value_key: an input key, representing the mean of Q(s, a) for every action `a` at state `s`.
        prior_action_value_key: an input key, representing the prior of Q(s, a) for every action `a` at state `s`.
        action_count_key: an input key, representing the number of times action `a` is selected at state `s`.
        action_value_under_uncertainty_key: an output key, representing the output estimate value using PUCT

    """

    def __init__(
        self,
        cpuct: float = 0.5,
        action_value_under_uncertainty_key: NestedKey = "action_value_under_uncertainty",
        action_value_key: NestedKey = "action_value",
        prior_action_value_key: NestedKey = "prior_action_value",
        action_count_key: NestedKey = "action_count",
    ):
        self.in_keys = [action_value_key, action_count_key, prior_action_value_key]
        self.out_keys = [action_value_under_uncertainty_key]
        super().__init__()
        self.cpuct = cpuct
        self.action_value_key = action_value_key
        self.prior_action_value_key = prior_action_value_key
        self.action_count_key = action_count_key
        self.action_value_under_uncertainty_key = action_value_under_uncertainty_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        n_sa = tensordict[self.action_count_key]
        p_sa = tensordict[self.prior_action_value_key]
        x_hat = tensordict[self.action_value_key]

        # we will always add 1, to avoid zero U values in the first visit of the node. See:
        # https://ai.stackexchange.com/questions/25451/how-does-alphazeros-mcts-work-when-starting-from-the-root-node
        # for a discussion on this topic.
        # TODO: investigate MuZero paper, AlphaZero paper and Bandit based monte-carlo planning to understand what
        # is the right implementation. Also check this discussion:
        # https://groups.google.com/g/computer-go-archive/c/K9XHb64JSqU
        n = torch.sum(n_sa, dim=-1) + 1
        u_sa = self.cpuct * p_sa * torch.sqrt(n) / (1 + n_sa)

        optimism_estimation = x_hat + u_sa
        tensordict[self.action_value_under_uncertainty_key] = optimism_estimation
        tensordict[self.action_value_under_uncertainty_key] = optimism_estimation

        return tensordict


class UcbSelectionPolicy(TensorDictModuleBase):
    """
    A policy to select an action in every node in the tree using UCB estimation.
    See Section 2.6 Upper-Confidence-Bound Action Selection on
    Sutton, Richard S., and Andrew G. Barto. 2018. “Reinforcement Learning: An Introduction (Second Edition).”
    http://incompleteideas.net/book/RLbook2020.pdf

    Args:
        action_value_key: The input key representing the mean of Q(s, a) for every action `a` at state `s`.
            Defaults to ``action_value``
        action_count_key: The input key representing the number of times action `a` is selected at state `s`.
          Defaults to ``action_count``
        action_value_under_uncertainty_key: The output key representing estimated action value.
    """

    def __init__(
        self,
        cucb: float = 2.0,
        action_value_under_uncertainty_key: NestedKey = "action_value_under_uncertainty",
        action_value_key: NestedKey = "action_value",
        action_count_key: NestedKey = "action_count",
    ):
        self.in_keys = [action_value_key, action_count_key]
        self.out_keys = [action_value_under_uncertainty_key]
        super().__init__()
        self.cucb = cucb
        self.action_value_key = action_value_key
        self.action_count_key = action_count_key
        self.action_value_under_uncertainty_key = action_value_under_uncertainty_key

    def forward(self, node: TensorDictBase) -> torch.Tensor:
        node = node.clone(False)
        x_hat = node[self.action_value_key]
        n_sa = node[self.action_count_key]
        mask = n_sa != 0
        n = torch.sum(n_sa)
        optimism_estimation = x_hat.clone()
        optimism_estimation[mask] = x_hat[mask] + self.cucb * torch.sqrt(
            torch.log(n) / n_sa[mask]
        )
        node[self.action_value_under_uncertainty_key] = optimism_estimation
        return node


class ActionExplorationModule(TensorDictModuleBase):
    def __init__(
        self,
        action_value_key: NestedKey = "action_value",
        action_value_under_uncertainty_key: NestedKey = "action_value_under_uncertainty",
        action_key: NestedKey = "action",
    ):
        self.in_keys = [action_value_key, action_value_under_uncertainty_key]
        self.out_keys = [action_key]
        super().__init__()
        self.action_value_key = action_value_under_uncertainty_key
        self.action_cnt_key = action_value_key
        self.action_key = action_key

    def forward(self, tensordict: TensorDictBase):
        tensordict = tensordict.clone(False)

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            tensordict[self.action_key] = self.explore_action(tensordict)
        elif exploration_type() == ExplorationType.MODE:
            tensordict[self.action_key] = self.get_greedy_action(tensordict)
        return tensordict

    def get_greedy_action(self, node: TensorDictBase) -> torch.Tensor:
        action_cnt_key = self.action_cnt_key
        action = torch.argmax(node[action_cnt_key], dim=-1)
        return torch.nn.functional.one_hot(action, node[action_cnt_key].shape[-1])

    def explore_action(self, node: TensorDictBase) -> torch.Tensor:
        action_value = node[self.action_value_key]
        max_value, _ = torch.max(action_value, dim=-1)
        action = torch.argmax(
            torch.rand_like(action_value) * (action_value == max_value)
        )
        return torch.nn.functional.one_hot(action, action_value.shape[-1])



class ActionExplorationModuleActionCheck(TensorDictModuleBase):
    def __init__(
        self,
        env: EnvBase,
        action_value_key: NestedKey = "action_value",
        action_value_under_uncertainty_key: NestedKey = "action_value_under_uncertainty",
        action_key: NestedKey = "action",
    ):
        self.in_keys = [action_value_key, action_value_under_uncertainty_key]
        self.out_keys = [action_key]
        super().__init__()
        self.action_value_key = action_value_under_uncertainty_key
        self.action_cnt_key = action_value_key
        self.action_key = action_key
        self.env = env

    def forward(self, tensordict: TensorDictBase):
        tensordict = tensordict.clone(False)

        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            tensordict[self.action_key] = self.explore_action(tensordict)
        elif exploration_type() == ExplorationType.MODE:
            tensordict[self.action_key] = self.get_greedy_action(tensordict)
        return tensordict

    def get_greedy_action(self, node: TensorDictBase) -> torch.Tensor:
        action_cnt_key = self.action_cnt_key
        action = torch.argmax(node[action_cnt_key], dim=-1)
        return torch.nn.functional.one_hot(action, node[action_cnt_key].shape[-1])

    def get_valid_action(self, node: TensorDictBase) -> torch.Tensor:
        cur_state = torch.argmax(node['observation']).clone().detach().item()
        action_space = self.env.P[cur_state]
        valid_action = set()
        for key, value in action_space.items():
            
            if value[0][1] != cur_state:
                valid_action.add(key)
            # print(key, value[0][1], cur_state, valid_action)

        return valid_action

        
    def explore_action(self, node: TensorDictBase) -> torch.Tensor:
        valid_action = list(self.get_valid_action(node))
        action_value = node[self.action_value_key]
        action_value_new = torch.zeros_like(action_value)
        action_value_new[valid_action] = action_value[valid_action]
        max_value, _ = torch.max(action_value_new, dim=-1)
        action = torch.argmax(
            torch.rand_like(action_value_new) * (action_value == max_value)
        )
        # print(torch.argmax(action), "action")
        return torch.nn.functional.one_hot(action, action_value_new.shape[-1])


class UpdateTreeStrategy(TensorDictModuleBase):
    """
    The strategy to update tree after each rollout. This class uses the given value estimator
    to compute a target value after each roll out and compute the mean of target values in the tree.

    It also updates the number of time nodes get visited in tree.

    Args:
        tree: A TensorDictMap that store stats of the tree.
        value_estimator: A ValueEstimatorBase that compute target value.
        action_key: A key in the rollout TensorDict to store the selected action.
        action_value_key: A key in the tree nodes that stores the mean of Q(s, a).
        action_count_key: A key in the tree nodes that stores the number of times nodes get visited.
    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        tree: TensorDictMap,
        value_estimator: Optional[ValueEstimatorBase] = None,
        rollout_key:  NestedKey = "rollout",
        action_key: NestedKey = "action",
        action_value_key: NestedKey = "action_value",
        action_count_key: NestedKey = "action_count",
        chosen_action_value_key: NestedKey = "chosen_action_value",
    ):
        self.in_keys = [rollout_key]
        self.out_keys = [rollout_key]
        super().__init__()
        self.tree = tree
        self.rollout_key = rollout_key
        self.action_key = action_key
        self.action_value_key = action_value_key
        self.action_count_key = action_count_key
        self.chosen_action_value = chosen_action_value_key
        self.value_estimator = value_estimator or self.get_default_value_network(tree)
        self.start_simulation()
        

    @staticmethod
    def get_default_value_network(tree: TensorDictMap) -> ValueEstimatorBase:
        # noinspection PyTypeChecker
        return TDLambdaEstimator(
            gamma=1.0,
            lmbda=1.0,
            value_network=MaxActionValue(tree),
            vectorized=False,  # Todo: use True instead and fix the error
        )
    
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        rollout = tensordict[self.rollout_key]
        print(rollout)
        return self.update(tensordict[self.rollout_key])

    def update(self, rollout: TensorDictBase) -> None:
        tree = self.tree
        action_count_key = self.action_count_key
        action_value_key = self.action_value_key
        # r = torch.max(rollout[("next", "reward")]).item()
        # usually time is along the last dimension (if envs are batched for instance)
        if len(rollout.batch_size) == 0:
            # Do not do anything here.
            return rollout
        steps = rollout.unbind(-1)

        value_estimator_input = rollout.unsqueeze(dim=0)
        target_value = self.value_estimator.value_estimate(value_estimator_input)
        target_value = target_value.squeeze(dim=0)

        target_values = target_value.unbind(rollout.ndim - 1)
        for idx in range(rollout.batch_size[-1]):
            state = steps[idx]
            node = tree[state]
            action = state[self.action_key]

            node[action_value_key] = safe_weighted_avg(
                node[action_count_key],
                node[action_value_key],
                action,
                target_values[idx],
            )
            node[action_count_key] += action
            tree[state] = node
        
        # return rollout so that we can train the alpha zero testing.
        return rollout # we are returning this to avoid returning the again during the alphazero coding.

    def start_simulation(self):
        self.tree.clear()

@dataclass
class MctsPolicyWithTreeUpdate(TensorDictSequential):
    """
    An implementation of MCTS algorithm.

    Args:
        tree_updated_strategy: a policy to update the tree after each simulation.
        expansion_strategy: a policy to initialize stats of a node at its first visit.
        selection_strategy: a policy to select action in each state
        exploration_strategy: a policy to exploration vs exploitation
    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        tree_update_strategy: UpdateTreeStrategy,
        expansion_strategy: ExpansionStrategy,
        selection_strategy: TensorDictModuleBase = UcbSelectionPolicy(),
        exploration_strategy: ActionExplorationModule = ActionExplorationModule(),
    ):
        super().__init__(
            tree_update_strategy, expansion_strategy, selection_strategy, exploration_strategy
        )  #
@dataclass
class MctsPolicy(TensorDictSequential):
    """
    An implementation of MCTS algorithm.

    Args:
        expansion_strategy: a policy to initialize stats of a node at its first visit.
        selection_strategy: a policy to select action in each state
        exploration_strategy: a policy to exploration vs exploitation
    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        expansion_strategy: ExpansionStrategy,
        selection_strategy: TensorDictModuleBase = UcbSelectionPolicy(),
        exploration_strategy: ActionExplorationModule = ActionExplorationModule(),
    ):
        super().__init__(
            expansion_strategy, selection_strategy, exploration_strategy
        )  #


@dataclass
class SimulatedSearchPolicy(TensorDictModuleBase):
    """
    A simulated search policy. In each step, it simulates `n` rollout of maximum steps of `max_steps`
    using the given policy and then choose the best action given the simulation results.

    Args:
        policy: a policy to select action in each simulation rollout.
        env: an environment to simulate a rollout
        num_simulation: the number of simulation
        max_steps: the max steps of each simulated rollout

    """

    def __init__(
        self,
        policy: MctsPolicy,
        tree_updater: UpdateTreeStrategy,
        env: EnvBase,
        num_simulation: int,
        max_steps: int,
    ):
        self.in_keys = policy.in_keys
        self.out_keys = policy.out_keys

        super().__init__()
        self.policy = policy
        self.tree_updater = tree_updater
        self.env = env
        self.num_simulation = num_simulation
        self.max_steps = max_steps
        self._rollout = [] # This is temporary, we will remove this later.

    def forward(self, tensordict: TensorDictBase):
        with torch.no_grad():
            self.tree_updater.start_simulation()

            for i in range(self.num_simulation):
                # print("Simulation", i)
                self._rollout.append(self.simulate(tensordict))
                # reset the env to avoid the side effect of the simulation reaching the end of the episode.
                out = self.env.reset()

            with set_exploration_type(ExplorationType.MODE):
                print(torch.argmax(tensordict['observation']))
                print(self.tree_updater.tree[tensordict]['action_count'])
                tensordict = self.policy(tensordict)
            return tensordict

    def simulate(self, tensordict: TensorDictBase):
        tensordict = tensordict.clone(False)
        rollout = self.env.rollout(
            max_steps=self.max_steps,
            policy=self.policy,
            tensordict=tensordict,
            auto_reset=False,
        )
        print(rollout['next']['observation'])
        print(rollout['next']['reward'])
        return self.tree_updater.update(rollout)
    
# @dataclass
# class SimulatedAlphaZeroSearchPolicy(TensorDictModuleBase):
#     """
#     A simulated search policy. In each step, it simulates `n` rollout of maximum steps of `max_steps`
#     using the given policy and then choose the best action given the simulation results.

#     Args:
#         policy: a policy to select action in each simulation rollout.
#         env: an environment to simulate a rollout
#         num_simulation: the number of simulation
#         max_steps: the max steps of each simulated rollout

#     """

#     def __init__(
#         self,
#         policy: MctsPolicy,
#         tree_updater: UpdateTreeStrategy,
#         env: EnvBase,
#         num_simulation: int,
#         max_steps: int,
#     ):
#         self.in_keys = policy.in_keys
#         self.out_keys = policy.out_keys

#         super().__init__()
#         self.policy = policy
#         self.tree_updater = tree_updater
#         self.env = env
#         self.num_simulation = num_simulation
#         self.max_steps = max_steps
#         self._store_rollout = [] # This is temporary, we will remove this later.
#         self.rollout = None

#     def forward(self, tensordict: TensorDictBase, ):
#         with torch.no_grad():
#             self.tree_updater.start_simulation()
#             self._store_rollout.clear()
#             for i in tqdm(range(self.num_simulation)):
#                 self.simulate(tensordict)

#             with set_exploration_type(ExplorationType.MODE):
#                 tensordict = self.policy(tensordict)
#             rollout = torch.cat(self._store_rollout, dim=-1)
#             self._store_rollout.clear()
#             self.rollout = rollout
#             return tensordict

#     def simulate(self, tensordict: TensorDictBase):
#         tensordict = tensordict.clone(False)
#         rollout = self.env.rollout(
#             max_steps=self.max_steps,
#             policy=self.policy,
#             tensordict=tensordict,
#             auto_reset=False,
#         )
#         self.tree_updater.update(rollout)
#         self._store_rollout.append(rollout)
        
