# Copyright 2021 Rishabh Jain. All Rights Reserved.
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
# ==============================================================================

"""JAX implementation of memory modules."""

import abc
from typing import Any, Callable, List, Optional, Tuple, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

_Array = chex.Array
_Fn = Callable[..., Any]

MEMORY_TAG = "clrs_memory_module"

MemoryState = NamedTuple


class MemoryModule(hk.Module):
    """Memory Module abstract base class."""

    def __init__(self, name: str):
        if not name.endswith(MEMORY_TAG):
            name = name + "_" + MEMORY_TAG
        super().__init__(name=name)

    @abc.abstractmethod
    def initial_state(
        self, batch_size: int, nb_nodes: int, hiddens: _Array, **kwargs
    ) -> MemoryState:
        """Memory module method to intiialize the state

        Returns:
          A memory state.
        """
        pass

    @abc.abstractmethod
    def __call__(
        self, nxt_hidden: _Array, cur_mem_state: MemoryState, **kwargs
    ) -> Tuple[_Array, MemoryState]:
        """Memory module inference step.

        Args:
          nxt_hidden: Next hidden state of nodes of the processor.
          current_state: Current state of the Memory Module.
          **kwargs: Extra kwargs.

        Returns:
          Output of memory module inference step as a 2-tuple of
          (node embeddings, next state of memory module).
        """
        pass


class LSTMModule(MemoryModule):
    """Memory Module wrapping for LSTM."""

    def __init__(self, hidden_size: int, name: str = "LSTM_wrapper"):
        """Initializes the LSTM wrapper and the module within it."""
        super().__init__(name)
        self.hidden_size = hidden_size
        self.lstm = hk.LSTM(hidden_size=hidden_size, name="hk_LSTM")

    def initial_state(
        self, batch_size: int, nb_nodes: int, hiddens: _Array, **kwargs
    ) -> hk.LSTMState:
        init_state = self.lstm.initial_state(batch_size * nb_nodes)
        init_state = jax.tree_util.tree_map(
            lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]), init_state
        )
        return init_state

    def __call__(
        self, nxt_hidden: _Array, cur_mem_state: hk.LSTMState, **kwargs
    ) -> Tuple[_Array, hk.LSTMState]:
        # lstm doesn't accept multiple batch dimensions (in our case, batch and
        # nodes), so we vmap over the (first) batch dimension.
        nxt_hidden, nxt_mem_state = jax.vmap(self.lstm)(nxt_hidden, cur_mem_state)
        return nxt_hidden, nxt_mem_state
