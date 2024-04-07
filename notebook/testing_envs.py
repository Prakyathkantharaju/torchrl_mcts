import torch
import sys
sys.path.append('../')
from mcts.tensordict_map import TensorDictMap
from mcts.mcts_policy import MctsPolicy, UpdateTreeStrategy, AlphaZeroExpansionStrategy, SimulatedSearchPolicy, ActionExplorationModuleActionCheck
from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv
from torchrl.modules import QValueActor, MLP
from torchrl.envs import GymEnv, TransformedEnv, Compose, DTypeCastTransform, StepCounter, RewardSum



# TODO: This test fails because the action value of reset state is getting changed between simulation 1 and
#   simulation 2, the general hypothesis is that the tensor changed in the dict when we explore a new action in this
#   state in UcbSelectionPolicy
torch.manual_seed(1)
# env = GymEnv("Taxi") 
env = TransformedEnv(
    GymEnv("Taxi"),
    Compose(
        DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]),
        StepCounter(),
    )
)

# print(env.P[100])




tree = TensorDictMap("observation")

value_module = QValueActor(
    MLP(in_features=500, out_features=6),
    in_keys='observation',
    action_space='categorical',
)
policy = SimulatedSearchPolicy(
    policy=MctsPolicy(
        expansion_strategy=AlphaZeroExpansionStrategy(
            tree=tree,
            value_module=value_module,
        ),
        exploration_strategy=ActionExplorationModuleActionCheck(env),
    ),
    tree_updater=UpdateTreeStrategy(tree),
    env=env,
    num_simulation=100,
    max_steps=50,
)


rollout = env.rollout(
    max_steps=100,
    policy=policy,
    break_when_any_done=False
)


for idx, v in enumerate(rollout[("next", "reward")].detach().numpy()):
    print(f"{idx}: {v}")