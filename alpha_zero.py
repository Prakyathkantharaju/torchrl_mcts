import torch
from torchrl.modules import QValueActor
from torchrl.envs import GymEnv, TransformedEnv, Compose, DTypeCastTransform, StepCounter
from torchrl.objectives import DQNLoss
from mcts.tensordict_map import TensorDictMap
from mcts.mcts_policy import MctsPolicy, UpdateTreeStrategy, AlphaZeroExpansionStrategy, UcbSelectionPolicy, SimulatedAlphaZeroSearchPolicy, ZeroExpansion, SimulatedSearchPolicy

from warnings import filterwarnings
filterwarnings("ignore")


def make_q_value(num_observation, num_action, action_space):
    net = torch.nn.Linear(num_observation, num_action)
    qvalue_module = QValueActor(net, in_keys=["observation"], action_space=action_space)
    return qvalue_module



env = TransformedEnv(
    #GymEnv("CliffWalking-v0"),
    GymEnv("FrozenLake-v1", map_name="4x4", is_slippery=False, desc=None),
    Compose(StepCounter())
    # Compose(
    #     DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32, in_keys=["observation"]), 
    #     StepCounter(),
    # )
)
# qvalue_module = make_q_value(env.observation_spec["observation"].shape[-1], env.action_spec.shape[-1], env.action_spec)
# qvalue_module(env.reset())

# loss_module = DQNLoss(qvalue_module, action_space=env.action_spec)

# tree = TensorDictMap(["observation", "step_count"])

tree = TensorDictMap("observation")
policy = SimulatedSearchPolicy(
    policy=MctsPolicy(
        expansion_strategy=ZeroExpansion(
            tree=tree, num_action=env.action_spec.shape[-1]
        ),
    ),
    tree_updater=UpdateTreeStrategy(tree, num_action=env.action_spec.shape[-1]),
    env=env,
    num_simulation=100,
    max_steps=3,
)

"""
policy = SimulatedSearchPolicy(
    policy=MctsPolicy(
        expansion_strategy=ZeroExpansion(tree=TensorDictMap("observation"), num_action=env.action_spec.shape[-1]),
        selection_strategy=UcbSelectionPolicy(),
    ),
    tree_updater=UpdateTreeStrategy(tree),
    env=env,
    num_simulation=10,
    max_steps=10,
)
"""
state = env.reset()
print(state['observation'])
for i in range(10):
    action = policy(state)
    state = env.step(action)
    print(state['action'], 'action after onehot')
    print(state['action_value'], 'action value at output')
    #print(state['observation'])
    #print(state['action_value'])
    #print(state['done'])
    if state['done']:
        break

# rollout = policy.rollout # TODO: This need to be updated and compapatible with torchrl methods ( Trained and collector )
# optimizer = torch.optim.Adam(qvalue_module.parameters(), lr=1e-7)

# # TODO: This needs to be in the trainer class torchrl.Trainer
# for i in range(10):
#     if i  == 1:
#         # change the learning rate
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-9 # update the learning rate because the loss is not decreasing.

#     # updated the qvalue module and the loss backpropagation
#     optimizer.zero_grad()
#     losses_td = loss_module(rollout)
#     loss_components = (item for key, item in losses_td.items() if key.startswith("loss"))
#     loss = sum(loss_components)
#     loss.backward()

#     for name, param in qvalue_module.named_parameters():
#         if param.grad is not None:
#             print(f"Gradient for {name}: {param.grad.abs().sum().item()}")
#     optimizer.step()
#     print(f'Loss: {loss.item()}')



#     # Next step prediction
#     next_state = env.step(action_dict)

#     # action for the next state
#     action_dict = policy(next_state)
