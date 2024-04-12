import torch
from typing import Sequence, Optional
from torchrl.data.tensor_specs import TensorSpec

from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase, NestedKey
from torchrl.envs import EnvBase
from tensordict.nn import TensorDictSequential
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec

class SimTransform(Transform):
    def __init__(self, 
            base_env: EnvBase,
            policy: Optional[TensorDictSequential] = None,
            number_of_simulations: int = 1,
            sim_key: NestedKey = "rollout",
            max_steps: Optional[int] = 200):
        super().__init__()
        self.base_env = base_env
        self.base_env.reset()
        self.policy = policy
        self.nos = number_of_simulations
        self.sim_key = sim_key
        self.max_steps = max_steps
    
    
    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        # This is where we are doing a simulated rollout. I will update the rollout the data 
        rollout = []
        for i in range(self.nos):
            rollout.append(self.base_env.rollout(policy = self.policy,
                                                           max_steps = self.nos,
                                                           tensordict = tensordict,
                                                           auto_reset = False))
            tensordict[self.sim_key] = torch.stack(rollout, 0)
        return tensordict
