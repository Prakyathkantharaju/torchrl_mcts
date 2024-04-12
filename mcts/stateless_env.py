from __future__ import annotations
import warnings
from copy import copy
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
)
import torch
# from torchrl.envs import EnvBase
from typing import Sequence
from torchrl.envs import GymEnv, EnvBase
from tensordict import TensorDictBase, NestedKey, TensorDict
from torchrl.envs.transforms import Transform, TransformedEnv
from torchrl.data.tensor_specs import CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec, OneHotDiscreteTensorSpec, DiscreteBox


import warnings
warnings.filterwarnings("ignore")



class StatelessGymEnv(GymEnv):
    def __init__(self, 
                 env_name: str,
                 state_key: Sequence[NestedKey] | str = "state", 
                 copy_env: bool = True,
                 get_state: Optional[Callable] = None,
                 set_state: Optional[Callable] = None,  
                 rollout_state: Optional[bool] = True, 
                 *args, **kwargs) -> None:
        """
        Stateful env to stateless env transform. Provides option to override the _copy_state method to provide custom state copying or None if the env is already stateless.

        Args:
            env (GymEnv): Stateful env to transform to stateless.  
            state_key (Sequence[NestedKey] | None, optional): Key to store the state in the tensordict. Defaults to None.
            in_keys (Sequence[NestedKey] | None, optional): Keys to select from the tensordict for the input of state. Defaults to None.
            out_keys (Sequence[NestedKey] | None, optional): Keys to select from the tensordict for the output of state. Defaults to None.
            method_of_state (str | None, optional): Method to copy the state. Defaults to "copy". If None is provider then the env is assumed to be stateless and not transform is done to override the state of the env.
        """
        # Not sure if the user can change this after instantiation ( will change to self._in_keys and self._out_keys, if it's a problem )
        self._state_key = state_key
        self._copy_env = copy_env
        self.get_state = get_state if get_state is not None else self._get_state
        self.set_state = set_state if set_state is not None else self._set_state
        self.if_rollout = rollout_state
        # the state function have to initialized before the parent class initialize to ensure that the
        super().__init__(env_name=env_name, *args, **kwargs)
        self._current_state = self.get_state(self.env)

    def _get_state(self, env: Any) -> Any:
        # default lagic will work for Taxi and frozenlake
        return env.s 

    def _set_state(self, env: Any, state: torch.Tensor) -> Any:
        # default lagic will work for Taxi and frozenlake
        env.s = state.item()
        return env

    def force_state_update(self, tensordict: TensorDictBase) -> TensorDictBase:
        state = tensordict[self._state_key] if self._state_key in tensordict.keys() else self._current_state  # This is the default behavior, will replace in future.
        env = self.set_state(self.env, state)
        tensordict[self._state_key] = self.get_state(env)
        return tensordict

    

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        env = copy(self.env) if self._copy_env else self.env
        state = tensordict[self._state_key] if self._state_key in tensordict.keys() else self._current_state  # This is the default behavior, will replace in future.
        env = self.set_state(env, state)
        next_tensordict = super()._step(tensordict)
        next_tensordict[self._state_key] = self.get_state(env)
        self.env = copy(env)
        del env
        if 'rollout' in tensordict.keys():
            next_tensordict['rollout'] = tensordict['rollout']
        return next_tensordict
    
    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        out_tensordict = super()._reset(tensordict, **kwargs)
        state = self.get_state(self.env)
        out_tensordict[self._state_key] = state
        if self.if_rollout and 'rollout' not in out_tensordict.keys():
            #TODO: Need a better way here, I am create a new "temp" rollout filled with zeros so that "rollout" key is present in the out_tensordict
            out_tensordict['rollout'] = self.create_tensordict_from_composite_spec(self.output_spec)
        return out_tensordict



    def create_tensordict_from_composite_spec(self, spec: CompositeSpec):
        tensordict = dict()  # Change device as needed

        for key, ts in spec.items():
            if isinstance(ts, CompositeSpec):
                for subkey, subts in ts.items():
                    tensor = self.initialize_tensor(subts)
                    tensordict[subkey] = tensor
            else:
                tensor = self.initialize_tensor(ts)
                tensordict[key] = tensor
        return TensorDict.from_dict(tensordict)
    
    def initialize_tensor(self, ts):
        if isinstance(ts, DiscreteTensorSpec) and ts.dtype is torch.bool:
            return torch.randint(0, 2, ts.shape, dtype=ts.dtype, device=ts.device)
        elif isinstance(ts, UnboundedContinuousTensorSpec):
            return torch.randn(ts.shape, dtype=ts.dtype, device=ts.device)
        elif isinstance(ts, OneHotDiscreteTensorSpec):
            data = torch.zeros(ts.shape, dtype=ts.dtype, device=ts.device)
            return data
        else:
            raise ValueError(f"Unhandled tensor spec type: {type(ts)}")


if __name__ == "__main__":


    def get_state(env: Any) -> Any:
        # This is where you will add logic to get the state of the env. This is usefull, because the in some case the observation is different from state. But In stateless env you need state to be absolute with anything hidden.
        return env.s 

    def set_state(env: Any, state: torch.Tensor) -> Any:
        env.s = state.item()
        return env


    env = StatelessGymEnv('Taxi-v3', get_state=get_state, set_state=set_state)

    # print(env.rollout(max_steps=1))


