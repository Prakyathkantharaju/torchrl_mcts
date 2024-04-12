
from typing import Sequence

from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase, NestedKey
from torchrl.envs import EnvBase
from tensordict import  TensorDictSequential

class SimTransform(Transform):
    def __init__(self, 
                 env: EnvBase,
                 policy: TensorDictSequential,
                 number_of_simulations: int,
                 sim_key: NestedKey,
                 max_steps: int,
                 in_keys: Sequence[NestedKey] | None = None, out_keys: Sequence[NestedKey] | None = None):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.env = env
        self.policy = policy
        self.nos = number_of_simulations
        self.sim_key = sim_key
        self.max_steps = max_steps
    
    
    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        # This is where we are doing a simulated rollout. I will update the rollout the data 
        for i in range(self.nos):
            tensordict[self.sim_key][i] = self.env.rollout(policy = self.policy,
                                                           max_steps = self.nos,
                                                           tensordict = tensordict,
                                                           auto_reset = True)
        return tensordict
