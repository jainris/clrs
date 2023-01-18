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
from clrs._src.external_memory import NeuralQueueCell
from clrs._src.external_memory import NeuralDequeCell
from clrs._src.external_memory import NeuralStackControllerInterface
from clrs._src.external_memory import NeuralStackState

_Array = jnp.ndarray
_Fn = Callable[..., Any]

MEMORY_TAG = "clrs_memory_module"

MemoryState = NamedTuple

SMALL_NUMBER = 1e-6


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
    controller_hidden_state: Optional[_Array]
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
        batch_size = nxt_hidden.shape[0]
        nb_nodes = nxt_hidden.shape[1]
        rest_shape = nxt_hidden.shape[2:]
        nxt_hidden = nxt_hidden.reshape([batch_size * nb_nodes, *rest_shape])
        nxt_hidden, nxt_mem_state = self.call_single_node(nxt_hidden, cur_mem_state)
        nxt_hidden = nxt_hidden.reshape([batch_size, nb_nodes, *rest_shape])
        return nxt_hidden, nxt_mem_state


class MLPStackMemory(MemoryModule):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        initialize_class: bool = True,
        name: str = "neural_stack_mlp",
    ):
        super().__init__(name=name)
        if initialize_class:
            self._output_size = output_size
            self._embedding_size = embedding_size
            self._memory_size = memory_size

            self.stack = NeuralStackCell(
                embedding_size=embedding_size,
                memory_size=memory_size,
                num_read_heads=1,
                num_write_heads=1,
            )

            self.push_proj = hk.Linear(output_size=1)
            self.pop_proj = hk.Linear(output_size=1)
            self.value_proj = hk.Linear(output_size=self._embedding_size)
            self.output_proj = hk.Linear(output_size=self._output_size)

    def initial_state(
        self, batch_size: int, nb_nodes: int, hiddens: _Array, **kwargs
    ) -> NeuralStackMemoryModuleState:
        init_stack_state = self.stack.initial_state(batch_size * nb_nodes)
        init_read_values = jnp.zeros(
            shape=[
                batch_size * nb_nodes,
                self.stack._num_read_heads,
                self._embedding_size,
            ]
        )
        init_controller = None  # MLP does not depend on previous state

        init_state = NeuralStackMemoryModuleState(
            memory_values=init_stack_state.memory_values,
            read_strengths=init_stack_state.read_strengths,
            write_mask=init_stack_state.write_mask,
            controller_hidden_state=init_controller,
            read_values=init_read_values,
        )
        return init_state

    def get_controller_shape(self, batch_size: int):
        return (
            # push_strengths,
            [batch_size, self.stack._num_write_heads, 1, 1],
            # pop_strengths
            [batch_size, self.stack._num_write_heads, 1, 1],
            # write_values
            [batch_size, self.stack._num_write_heads, self._embedding_size],
        )

    def call_controller(
        self,
        nxt_hidden: _Array,
        batch_size: int,
    ):
        push_strengths = jax.nn.sigmoid(self.push_proj(nxt_hidden))
        pop_strengths = jax.nn.sigmoid(self.pop_proj(nxt_hidden))

        write_values = jnp.tanh(self.value_proj(nxt_hidden))

        projected_outputs = [
            push_strengths,
            pop_strengths,
            write_values,
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
        (push_strengths, pop_strengths, write_values,) = self.call_controller(
            nxt_hidden=nxt_hidden,
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
            controller_hidden_state=None,
            read_values=nxt_read_values,
        )
        nxt_read_values = jnp.reshape(nxt_read_values, (batch_size, -1))
        nxt_read_values = self.output_proj(nxt_read_values)
        return nxt_read_values, new_state

    def __call__(
        self, nxt_hidden: _Array, cur_mem_state: NeuralStackMemoryModuleState, **kwargs
    ) -> Tuple[_Array, NeuralStackMemoryModuleState]:
        batch_size = nxt_hidden.shape[0]
        nb_nodes = nxt_hidden.shape[1]
        rest_shape = nxt_hidden.shape[2:]
        if cur_mem_state is None:
            cur_mem_state = self.initial_state(
                batch_size=batch_size, nb_nodes=nb_nodes, hiddens=nxt_hidden
            )
        nxt_hidden = nxt_hidden.reshape([batch_size * nb_nodes, *rest_shape])
        nxt_read_values, nxt_mem_state = self.call_single_node(
            nxt_hidden, cur_mem_state
        )
        nxt_read_values = nxt_read_values.reshape(
            [batch_size, nb_nodes, self._output_size]
        )
        return nxt_read_values, nxt_mem_state


class MLPQueueMemory(MLPStackMemory):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        name: str = "neural_queue_mlp",
    ):
        super().__init__(
            output_size=output_size,
            embedding_size=embedding_size,
            memory_size=memory_size,
            initialize_class=False,
            name=name,
        )
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size

        self.stack = NeuralQueueCell(
            memory_size=memory_size, embedding_size=embedding_size
        )

        self.push_proj = hk.Linear(output_size=1)
        self.pop_proj = hk.Linear(output_size=1)
        self.value_proj = hk.Linear(output_size=self._embedding_size)
        self.output_proj = hk.Linear(output_size=self._output_size)


class MLPDequeMemory(MLPStackMemory):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        name: str = "neural_deque_mlp",
    ):
        super().__init__(
            output_size=output_size,
            embedding_size=embedding_size,
            memory_size=memory_size,
            initialize_class=False,
            name=name,
        )
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size

        self.stack = NeuralDequeCell(
            memory_size=memory_size, embedding_size=embedding_size
        )

        self.push_proj = hk.Linear(output_size=2)
        self.pop_proj = hk.Linear(output_size=2)
        self.value_proj = hk.Linear(output_size=2 * self._embedding_size)
        self.output_proj = hk.Linear(output_size=self._output_size)


class PriorityQueueState(MemoryState):
    memory_values: _Array
    # read_strengths: _Array
    write_mask: _Array
    bias_mask: _Array


class PriorityQueue(MemoryModule):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        nb_heads: int,
        aggregation_technique: str = "max",
        name: str = "neural_priority_queue",
    ):
        super().__init__(name=name)
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size
        self.nb_heads = nb_heads
        self.aggregation_technique = aggregation_technique

    def initial_state(self, batch_size: int, **kwargs) -> PriorityQueueState:
        memory_values = jnp.zeros((batch_size, self._memory_size, self._embedding_size))
        write_mask = jax.nn.one_hot(
            [0] * batch_size,
            num_classes=self._memory_size,
            dtype=jnp.float32,
        )  # [B, S]
        bias_mask = jnp.zeros((batch_size, self._memory_size))
        return PriorityQueueState(
            memory_values=memory_values, write_mask=write_mask, bias_mask=bias_mask
        )

    def __call__(
        self, z: _Array, prev_state: PriorityQueueState, **kwargs
    ) -> Tuple[_Array, PriorityQueueState]:
        # z.shape: [B, N, _]
        batch_size, nb_nodes, nb_z_fts = z.shape
        if prev_state is None:
            prev_state = self.initial_state(batch_size=batch_size)

        # push_proj = hk.Linear(output_size=1)
        # pop_proj = hk.Linear(output_size=1)
        value_proj = hk.Linear(output_size=self._embedding_size)
        output_proj = hk.Linear(output_size=self._output_size)

        # push_strengths = jax.nn.sigmoid(push_proj(z))
        # pop_strengths = jax.nn.sigmoid(pop_proj(z))

        write_values = value_proj(z.reshape(batch_size * nb_nodes, nb_z_fts))
        write_values = jnp.reshape(
            write_values, (batch_size, nb_nodes, self._embedding_size)
        )
        write_values = jnp.sum(write_values, axis=1)
        write_values = jnp.tanh(write_values)

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)

        att_1 = jnp.expand_dims(a_1(z), axis=-1)
        att_2 = jnp.expand_dims(a_2(prev_state.memory_values), axis=-1)  # [B, S, H, 1]

        # [B, H, N, 1] + [B, H, 1, S]  = [B, H, N, S]
        logits = jnp.transpose(att_1, (0, 2, 1, 3)) + jnp.transpose(att_2, (0, 2, 3, 1))

        # Masking out the unwritten memory cells
        # mem_idx = jnp.argmax(prev_state.write_mask, axis=-1)  # [B]
        # fill_ones_till = jax.vmap(
        #     lambda x: jnp.sum(jax.nn.one_hot(jnp.arange(x)), axis=0), 0, 0
        # )
        # bias_mat = (fill_ones_till(mem_idx) - 1) * 1e9  # [B, S]
        bias_mat = (prev_state.bias_mask - 1) * 1e9  # [B, S]
        bias_mat = jnp.expand_dims(bias_mat, 1)  # [B, 1, S]
        bias_mat = jnp.expand_dims(bias_mat, 1)  # [B, 1, 1, S]
        bias_mat = jnp.tile(bias_mat, (1, self.nb_heads, nb_nodes, 1))  # [B, H, N, S]

        # bias_mat = jnp.zeros((nb_nodes, self.nb_heads, batch_size, self._memory_size)) # [N, H, B, S]
        # bias_mat[:, :, mem_idx:] = -1e9

        coefs = jax.nn.softmax(
            jax.nn.leaky_relu(logits) + bias_mat, axis=-1
        )  # [B, H, N, S]
        # Projecting from the different heads into a single importance value
        coefs = jnp.transpose(coefs, (0, 2, 3, 1))  # [B, N, S, H]
        if self.nb_heads > 1:
            # Multi-head
            coefs = hk.Linear(output_size=1)(coefs)  # [B, N, S, 1]
            coefs = jnp.squeeze(coefs, axis=3)  # [B, N, S]
            coefs = jax.nn.softmax(
                jax.nn.leaky_relu(coefs) + bias_mat[:, 0], axis=-1
            )  # [B, N, S]
        else:
            # Single-head
            coefs = jnp.squeeze(coefs, axis=3)  # [B, N, S]

        if self.aggregation_technique == "max":
            max_att_idx = jnp.argmax(
                jnp.reshape(coefs, (batch_size * nb_nodes, self._memory_size)),
                axis=-1,
            )  # [B * N]
            # batch_idx = jnp.tile(jnp.arange(batch_size), nb_nodes)
            batch_idx = jnp.repeat(jnp.arange(batch_size), nb_nodes)
            output = prev_state.memory_values[batch_idx, max_att_idx]  # [B * N, F']
            output = jnp.reshape(output, (batch_size, nb_nodes, -1))  # [B, N, F']
        elif self.aggregation_technique == "weighted":
            val = prev_state.memory_values
            # val = jnp.expand_dims(prev_state.memory_values, axis=0) # [1, S, F']
            # val = jnp.tile(prev_state.memory_values, (batch_size, 1, 1, 1)) # [B, S, F']
            # print("val shape: {}".format(val.shape))
            output = jnp.matmul(coefs, val)  # [B, N, F']
        else:
            raise ValueError(
                "Unknown memory aggregation technique " + self.aggregation_technique
            )
        output = output_proj(output)

        new_memory_values = prev_state.memory_values + jnp.expand_dims(
            write_values, axis=1
        ) * jnp.expand_dims(prev_state.write_mask, axis=2)
        new_write_mask = jnp.roll(prev_state.write_mask, shift=1, axis=1)

        new_bias_mask = jnp.minimum((prev_state.bias_mask + prev_state.write_mask), 1)

        return output, PriorityQueueState(
            memory_values=new_memory_values,
            write_mask=new_write_mask,
            bias_mask=new_bias_mask,
        )


class PriorityQueueV1(MemoryModule):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        nb_heads: int,
        aggregation_technique: str = "max",
        name: str = "neural_priority_queue_v1",
    ):
        super().__init__(name=name)
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size
        self.nb_heads = nb_heads
        self.aggregation_technique = aggregation_technique

    def initial_state(self, batch_size: int, **kwargs) -> PriorityQueueState:
        memory_values = jnp.zeros((batch_size, self._memory_size, self._embedding_size))
        write_mask = jax.nn.one_hot(
            [0] * batch_size,
            num_classes=self._memory_size,
            dtype=jnp.float32,
        )  # [B, S]
        bias_mask = jnp.zeros((batch_size, self._memory_size))
        return PriorityQueueState(
            memory_values=memory_values, write_mask=write_mask, bias_mask=bias_mask
        )

    def __call__(
        self, z: _Array, prev_state: PriorityQueueState, **kwargs
    ) -> Tuple[_Array, PriorityQueueState]:
        # z.shape: [B, N, _]
        batch_size, nb_nodes, nb_z_fts = z.shape
        if prev_state is None:
            prev_state = self.initial_state(batch_size=batch_size)

        # push_proj = hk.Linear(output_size=1)
        # pop_proj = hk.Linear(output_size=1)
        value_proj = hk.Linear(output_size=self._embedding_size)
        output_proj = hk.Linear(output_size=self._output_size)

        # push_strengths = jax.nn.sigmoid(push_proj(z))
        # pop_strengths = jax.nn.sigmoid(pop_proj(z))

        write_values = value_proj(z.reshape(batch_size * nb_nodes, nb_z_fts))
        write_values = jnp.reshape(
            write_values, (batch_size, nb_nodes, self._embedding_size)
        )
        write_values = jnp.sum(write_values, axis=1)
        write_values = jnp.tanh(write_values)

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)

        att_1 = jnp.expand_dims(a_1(z), axis=-1)
        att_2 = jnp.expand_dims(a_2(prev_state.memory_values), axis=-1)  # [B, S, H, 1]

        # [B, H, N, 1] + [B, H, 1, S]  = [B, H, N, S]
        logits = jnp.transpose(att_1, (0, 2, 1, 3)) + jnp.transpose(att_2, (0, 2, 3, 1))

        # Masking out the unwritten memory cells
        # mem_idx = jnp.argmax(prev_state.write_mask, axis=-1)  # [B]
        # fill_ones_till = jax.vmap(
        #     lambda x: jnp.sum(jax.nn.one_hot(jnp.arange(x)), axis=0), 0, 0
        # )
        # bias_mat = (fill_ones_till(mem_idx) - 1) * 1e9  # [B, S]
        bias_mat = (prev_state.bias_mask - 1) * 1e9  # [B, S]
        bias_mat = jnp.expand_dims(bias_mat, 1)  # [B, 1, S]
        bias_mat = jnp.expand_dims(bias_mat, 1)  # [B, 1, 1, S]
        bias_mat = jnp.tile(bias_mat, (1, self.nb_heads, nb_nodes, 1))  # [B, H, N, S]

        # bias_mat = jnp.zeros((nb_nodes, self.nb_heads, batch_size, self._memory_size)) # [N, H, B, S]
        # bias_mat[:, :, mem_idx:] = -1e9

        coefs = jax.nn.softmax(
            jax.nn.leaky_relu(logits) + bias_mat, axis=-1
        )  # [B, H, N, S]
        # Projecting from the different heads into a single importance value
        coefs = jnp.transpose(coefs, (0, 2, 3, 1))  # [B, N, S, H]
        if self.nb_heads > 1:
            # Multi-head
            coefs = hk.Linear(output_size=1)(coefs)  # [B, N, S, 1]
            coefs = jnp.squeeze(coefs, axis=3)  # [B, N, S]
            coefs = jax.nn.softmax(
                jax.nn.leaky_relu(coefs) + bias_mat[:, 0], axis=-1
            )  # [B, N, S]
        else:
            # Single-head
            coefs = jnp.squeeze(coefs, axis=3)  # [B, N, S]

        # Aggregating the attention values
        coefs = jnp.sum(coefs, axis=1, keepdims=False)  # [B, S]
        coefs = jax.nn.softmax(coefs, axis=-1)  # [B, S]

        if self.aggregation_technique == "max":
            max_att_idx = jnp.argmax(
                coefs,
                axis=-1,
            )  # [B]
            max_att_idx = jnp.repeat(max_att_idx, nb_nodes)
            batch_idx = jnp.repeat(jnp.arange(batch_size), nb_nodes)
            output = prev_state.memory_values[batch_idx, max_att_idx]  # [B * N, F']
            output = jnp.reshape(output, (batch_size, nb_nodes, -1))  # [B, N, F']
        elif self.aggregation_technique == "weighted":
            val = prev_state.memory_values  # [B, S, F']
            output = jnp.matmul(jnp.expand_dims(coefs, axis=1), val)  # [B, 1, F']
            output = jnp.tile(output, (1, nb_nodes, 1))  # [B, N, F']
        else:
            raise ValueError(
                "Unknown memory aggregation technique " + self.aggregation_technique
            )
        output = output_proj(output)

        new_memory_values = prev_state.memory_values + jnp.expand_dims(
            write_values, axis=1
        ) * jnp.expand_dims(prev_state.write_mask, axis=2)
        new_write_mask = jnp.roll(prev_state.write_mask, shift=1, axis=1)

        new_bias_mask = jnp.minimum((prev_state.bias_mask + prev_state.write_mask), 1)

        return output, PriorityQueueState(
            memory_values=new_memory_values,
            write_mask=new_write_mask,
            bias_mask=new_bias_mask,
        )


class PriorityQueueStateV2(MemoryState):
    memory_values: _Array
    read_strengths: _Array
    write_mask: _Array


class PriorityQueueV2(MemoryModule):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        nb_heads: int,
        aggregation_technique: str = "max",
        name: str = "neural_priority_queue_v2",
    ):
        super().__init__(name=name)
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size
        self.nb_heads = nb_heads
        self.aggregation_technique = aggregation_technique

    def initial_state(self, batch_size: int, **kwargs) -> PriorityQueueStateV2:
        memory_values = jnp.zeros((batch_size, self._memory_size, self._embedding_size))
        read_strengths = jnp.zeros((batch_size, self._memory_size))
        write_mask = jax.nn.one_hot(
            [0] * batch_size,
            num_classes=self._memory_size,
            dtype=jnp.float32,
        )  # [B, S]
        return PriorityQueueStateV2(
            memory_values=memory_values,
            read_strengths=read_strengths,
            write_mask=write_mask,
        )

    def __call__(
        self, z: _Array, prev_state: PriorityQueueStateV2, **kwargs
    ) -> Tuple[_Array, PriorityQueueStateV2]:
        # z.shape: [B, N, F]
        batch_size, nb_nodes, nb_z_fts = z.shape
        if prev_state is None:
            prev_state = self.initial_state(batch_size=batch_size)

        push_proj = hk.Linear(output_size=1)
        pop_proj = hk.Linear(output_size=1)
        value_proj = hk.Linear(output_size=self._embedding_size)
        output_proj = hk.Linear(output_size=self._output_size)

        push_strengths = push_proj(
            jnp.reshape(z, (batch_size * nb_nodes, nb_z_fts))
        )  # [B * N, 1]
        push_strengths = jnp.reshape(
            push_strengths, (batch_size, nb_nodes, 1)
        )  # [B, N, 1]
        push_strengths = jnp.sum(push_strengths, axis=1)  # [B, 1]
        push_strengths = jax.nn.sigmoid(push_strengths)  # [B, 1]

        pop_strengths = pop_proj(
            jnp.reshape(z, (batch_size * nb_nodes, nb_z_fts))
        )  # [B * N, 1]
        pop_strengths = jnp.reshape(pop_strengths, (batch_size, nb_nodes))  # [B, N]
        pop_strengths_sum = jnp.sum(pop_strengths, axis=1)  # [B]
        # node_wise_pop_strengths = jnp.expand_dims(pop_strengths_sum, axis=1) # [B, 1]
        # node_wise_pop_strengths = (
        #     jnp.ones((batch_size, nb_nodes, 1))
        #     * jnp.expand_dims(node_wise_pop_strengths, axis=1)
        #     / nb_nodes
        # )  # [B, N, 1]
        pop_strengths = jax.nn.softmax(pop_strengths, axis=1)  # [B, N]
        pop_strengths = (
            jnp.expand_dims(pop_strengths_sum, axis=1) * pop_strengths
        )  # [B, N]
        node_wise_pop_strengths = jnp.expand_dims(pop_strengths, axis=2)  # [B, N, 1]

        write_values = value_proj(z.reshape(batch_size * nb_nodes, nb_z_fts))
        write_values = jnp.reshape(
            write_values, (batch_size, nb_nodes, self._embedding_size)
        )  # [B, N, F]
        write_values = jnp.sum(write_values, axis=1)
        write_values = jnp.tanh(write_values)  # [B, F']

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)

        att_1 = jnp.expand_dims(a_1(z), axis=-1)  # [B, N, H, 1]
        att_2 = jnp.expand_dims(a_2(prev_state.memory_values), axis=-1)  # [B, S, H, 1]

        # [B, H, N, 1] + [B, H, 1, S]  = [B, H, N, S]
        logits = jnp.transpose(att_1, (0, 2, 1, 3)) + jnp.transpose(att_2, (0, 2, 3, 1))

        # Masking out the unwritten memory cells
        unwritten_mask = prev_state.read_strengths < SMALL_NUMBER
        bias_mat = unwritten_mask * -1e9  # [B, S]
        bias_mat = jnp.expand_dims(bias_mat, 1)  # [B, 1, S]
        bias_mat = jnp.expand_dims(bias_mat, 1)  # [B, 1, 1, S]
        bias_mat = jnp.tile(bias_mat, (1, self.nb_heads, nb_nodes, 1))  # [B, H, N, S]

        coefs = jax.nn.softmax(
            jax.nn.leaky_relu(logits) + bias_mat, axis=-1
        )  # [B, H, N, S]
        # Projecting from the different heads into a single importance value
        coefs = jnp.transpose(coefs, (0, 2, 3, 1))  # [B, N, S, H]
        if self.nb_heads > 1:
            # Multi-head
            coefs = hk.Linear(output_size=1)(coefs)  # [B, N, S, 1]
            coefs = jnp.squeeze(coefs, axis=3)  # [B, N, S]
            coefs = jax.nn.softmax(
                jax.nn.leaky_relu(coefs) + bias_mat[:, 0], axis=-1
            )  # [B, N, S]
        else:
            # Single-head
            coefs = jnp.squeeze(coefs, axis=3)  # [B, N, S]

        new_read_strengths = prev_state.read_strengths
        # Calculating the proportion of each element popped
        # (and their contribute to individual outputs)
        if self.aggregation_technique == "max":
            max_att_idx = jnp.argmax(
                jnp.reshape(coefs, (batch_size * nb_nodes, self._memory_size)),
                axis=-1,
            )  # [B * N]
            max_att_idx = jnp.reshape(max_att_idx, (batch_size, nb_nodes))  # [B, N]

            one_hot_max_att = jax.nn.one_hot(
                max_att_idx,
                num_classes=self._memory_size,
            )  # [B, N, S]
            pop_requested = node_wise_pop_strengths * one_hot_max_att  # [B, N, S]
        elif self.aggregation_technique == "weighted":
            pop_requested = node_wise_pop_strengths * coefs  # [B, N, S]
        else:
            raise ValueError(
                "Unknown memory aggregation technique " + self.aggregation_technique
            )
        total_pop_requested = jnp.sum(
            pop_requested,
            axis=1,
        )  # [B, S]
        recp_total_pop_requested = 1 / (total_pop_requested + SMALL_NUMBER)
        pop_proportions = pop_requested * jnp.expand_dims(
            recp_total_pop_requested, axis=1
        )  # [B, N, S]

        pop_given = jnp.minimum(
            prev_state.read_strengths, total_pop_requested
        )  # [B, S]
        output_coefs = pop_proportions * jnp.expand_dims(pop_given, axis=1)  # [B, N, S]
        output = jnp.expand_dims(output_coefs, axis=3) * jnp.expand_dims(
            prev_state.memory_values, axis=1
        )  # [B, N, S, F']
        output = jnp.sum(output, axis=2)  # [B, N, F']
        new_read_strengths -= pop_given
        output = output_proj(output)

        new_memory_values = prev_state.memory_values + jnp.expand_dims(
            write_values, axis=1
        ) * jnp.expand_dims(prev_state.write_mask, axis=2)
        unwritten_mask = new_read_strengths < SMALL_NUMBER
        new_read_strengths *= 1 - unwritten_mask
        new_read_strengths = new_read_strengths + push_strengths * prev_state.write_mask

        unwritten_mask = new_read_strengths < SMALL_NUMBER
        get_next_write_places = jax.vmap(lambda x: jnp.flatnonzero(x, size=1), 0, 0)
        next_write_places = get_next_write_places(unwritten_mask)  # [B, 1]
        next_write_places = jnp.squeeze(next_write_places, axis=-1)  # [B]
        new_write_mask = jax.nn.one_hot(
            next_write_places, num_classes=self._memory_size
        )  # [B, S]
        # def breakpoint_if_nonfinite(x):
        #     is_finite = jnp.isfinite(x).all()
        #     def true_fn(x):
        #         pass
        #     def false_fn(x):
        #         jax.debug.breakpoint()
        #     jax.lax.cond(is_finite, true_fn, false_fn, x)
        # breakpoint_if_nonfinite(output)

        return output, PriorityQueueStateV2(
            memory_values=new_memory_values,
            write_mask=new_write_mask,
            read_strengths=new_read_strengths,
        )
