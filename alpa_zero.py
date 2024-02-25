
import torch

from tensordict.nn import TensorDictModule

from torchrl.modules import QValueActor

from torchrl.envs import GymEnv, TransformedEnv, Compose, DTypeCastTransform, StepCounter

from torchrl.objectives import DQNLoss



def make_q_value(num_observation, num_action, action_space):
    net = torch.nn.Linear(num_observation, num_action)
    qvalue_module = QValueActor(net, in_keys=["observation"], action_space=action_space)
    return qvalue_module


env = TransformedEnv(
    GymEnv("CliffWalking-v0"),
    Compose(
        DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]), 
        StepCounter(),
    )
)
qvalue_module = make_q_value(env.observation_spec["observation"].shape[-1], env.action_spec.shape[-1], env.action_spec)
qvalue_module(env.reset())

loss_module = DQNLoss(qvalue_module, action_space=env.action_spec)

from mcts.tensordict_map import TensorDictMap
from mcts.mcts_policy import SimulatedSearchPolicy, MctsPolicy, UpdateTreeStrategy, AlphaZeroExpansionStrategy, PucbSelectionPolicy, SimulatedAlphaZeroSearchPolicy

tree = TensorDictMap(["observation", "step_count"])


policy = SimulatedAlphaZeroSearchPolicy(
    policy=MctsPolicy(
        expansion_strategy=AlphaZeroExpansionStrategy(value_module=qvalue_module, tree=tree),
        selection_strategy=PucbSelectionPolicy(),
    ),
    tree_updater=UpdateTreeStrategy(tree),
    env=env,
    num_simulation=10,
    max_steps=100,
)

res = policy(env.reset())
rollout = policy.rollout
optimizer = torch.optim.Adam(qvalue_module.parameters(), lr=1e-7)

for i in range(10):
    if i == 2:  # Change the learning rate after 2 iterations because the loss drops by x10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
    optimizer.zero_grad()

    losses_td = loss_module(rollout)
    loss_components = (item for key, item in losses_td.items() if key.startswith("loss"))
    loss = sum(loss_components)
    loss.backward()

    for name, param in qvalue_module.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.abs().sum().item()}")
    optimizer.step()
    print(f'Loss: {loss.item()}')

    # Generate new rollout for the next iteration
    res = env.rollout(max_steps=10, policy=policy)
    print(policy.rollout.shape)
    rollout = policy.rollout