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

from clrs._src.external_memory import NeuralStackCell
from clrs._src.external_memory import NeuralStackControllerInterface
from clrs._src.external_memory import NeuralStackState

_Array = jnp.ndarray
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


class NeuralStackMemoryModuleState(NamedTuple):
    # stack_state: NeuralStackState
    memory_values: _Array
    read_strengths: _Array
    write_mask: _Array
    controller_hidden_state: _Array
    read_values: _Array


class NeuralStackMemoryModule(MemoryModule):
    """Memory Module wrapping for neural Stack."""

    def __init__(
        self,
        num_units: int,
        embedding_size: int,
        memory_size: int,
        name: str = "neural_stack_rnn",
    ):
        super().__init__(name=name)
        self._num_units = num_units
        self._embedding_size = embedding_size
        self._memory_size = memory_size

        self.stack = NeuralStackCell(
            embedding_size=embedding_size,
            memory_size=memory_size,
            num_read_heads=1,
            num_write_heads=1,
        )
        self.rnn_controller = hk.VanillaRNN(
            hidden_size=embedding_size, name="neural_stack_controller"
        )

        self.input_projection = hk.Linear(output_size=self._embedding_size)
        self.push_proj = hk.Linear(output_size=1)
        self.pop_proj = hk.Linear(output_size=1)
        self.value_proj = hk.Linear(output_size=self._embedding_size)
        self.output_proj = hk.Linear(output_size=self._embedding_size)

    def initial_state(
        self, batch_size: int, nb_nodes: int, hiddens: _Array, **kwargs
    ) -> NeuralStackMemoryModuleState:
        init_stack_state = self.stack.initial_state(batch_size * nb_nodes)
        init_read_values = jnp.zeros(
            shape=[batch_size * nb_nodes, 1, self._embedding_size]
        )
        init_controller = self.rnn_controller.initial_state(batch_size * nb_nodes)

        init_state = NeuralStackMemoryModuleState(
            memory_values=init_stack_state.memory_values,
            read_strengths=init_stack_state.read_strengths,
            write_mask=init_stack_state.write_mask,
            controller_hidden_state=init_controller,
            read_values=init_read_values,
        )

        # init_state = jax.tree_util.tree_map(
        #     lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, *x.shape[1:]]),
        #     init_state,
        # )
        # init_state = jax.tree_util.tree_map(
        #     lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
        #     init_state,
        # )
        return init_state

    def get_controller_shape(self, batch_size: int):
        return (
            # push_strengths,
            [batch_size, 1, 1, 1],
            # pop_strengths
            [batch_size, 1, 1, 1],
            # write_values
            [batch_size, 1, self._embedding_size],
            # outputs
            [batch_size, 1, self._embedding_size],
            # state
            [batch_size, self._num_units],
        )

    def call_controller(
        self,
        nxt_hidden: _Array,
        controller_state: _Array,
        read_values: _Array,
        batch_size: int,
    ):
        controller_input = jnp.concatenate(
            [
                jnp.reshape(nxt_hidden, newshape=[batch_size, -1]),
                jnp.reshape(read_values, newshape=[batch_size, -1]),
            ],
            axis=1,
        )
        rnn_input = jnp.tanh(self.input_projection(controller_input))

        (rnn_output, state) = self.rnn_controller(rnn_input, controller_state)

        push_strengths = jax.nn.sigmoid(self.push_proj(rnn_output))
        pop_strengths = jax.nn.sigmoid(self.pop_proj(rnn_output))

        write_values = jnp.tanh(self.value_proj(rnn_output))
        outputs = jnp.tanh(self.output_proj(rnn_output))

        projected_outputs = [
            push_strengths,
            pop_strengths,
            write_values,
            outputs,
            state,
        ]

        next_state = [
            jnp.reshape(output, newshape=output_shape)
            for output, output_shape in zip(
                projected_outputs, self.get_controller_shape(batch_size)
            )
        ]
        return next_state

    def call_single_node(
        self, nxt_hidden: _Array, cur_mem_state: NeuralStackMemoryModuleState
    ):
        batch_size = nxt_hidden.shape[0]
        (
            push_strengths,
            pop_strengths,
            write_values,
            outputs,
            state,
        ) = self.call_controller(
            nxt_hidden=nxt_hidden,
            controller_state=cur_mem_state.controller_hidden_state,
            read_values=cur_mem_state.read_values,
            batch_size=batch_size,
        )

        stack_input = NeuralStackControllerInterface(
            write_values=write_values,
            push_strengths=push_strengths,
            pop_strengths=pop_strengths,
        )
        cur_stack_state = NeuralStackState(
            memory_values=cur_mem_state.memory_values,
            read_strengths=cur_mem_state.read_strengths,
            write_mask=cur_mem_state.write_mask,
        )

        nxt_read_values, nxt_stack_state = self.stack(stack_input, cur_stack_state)

        new_state = NeuralStackMemoryModuleState(
            memory_values=nxt_stack_state.memory_values,
            read_strengths=nxt_stack_state.read_strengths,
            write_mask=nxt_stack_state.write_mask,
            controller_hidden_state=state,
            read_values=nxt_read_values,
        )
        return outputs, new_state

    def __call__(
        self, nxt_hidden: _Array, cur_mem_state: NeuralStackMemoryModuleState, **kwargs
    ) -> Tuple[_Array, NeuralStackMemoryModuleState]:
        # nxt_hidden, nxt_mem_state = jax.vmap(self.call_single_node)(
        #     nxt_hidden, cur_mem_state
        # )
        batch_size = nxt_hidden.shape[0]
        nb_nodes = nxt_hidden.shape[1]
        rest_shape = nxt_hidden.shape[2:]
        nxt_hidden = nxt_hidden.reshape([batch_size * nb_nodes, *rest_shape])
        nxt_hidden, nxt_mem_state = self.call_single_node(nxt_hidden, cur_mem_state)
        nxt_hidden = nxt_hidden.reshape([batch_size, nb_nodes, *rest_shape])
        return nxt_hidden, nxt_mem_state

    # def call_single_node(
    #     self,
    #     nxt_hidden: _Array,
    #     controller_hidden_state: _Array,
    #     read_values: _Array,
    #     memory_values: jnp.ndarray,
    #     read_strengths: jnp.ndarray,
    #     write_mask: jnp.ndarray,
    # ):
    #     pass

    # def __call__(
    #     self, nxt_hidden: _Array, cur_mem_state: NeuralStackMemoryModuleState, **kwargs
    # ) -> Tuple[_Array, NeuralStackMemoryModuleState]:
    #     nxt_hidden, nxt_mem_state = jax.vmap(self.call_single_node)(
    #         nxt_hidden,
    #         cur_mem_state.controller_hidden_state,
    #         cur_mem_state.read_values,
    #         cur_mem_state.stack_state.memory_values,
    #         cur_mem_state.stack_state.read_strengths,
    #         cur_mem_state.stack_state.write_mask,
    #     )
    #     return nxt_hidden, nxt_mem_state
