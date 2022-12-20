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

"""JAX implementation of the external memory (sub)modules."""

import abc
from typing import Any, Callable, List, Optional, Tuple, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class NeuralStackState(NamedTuple):
    """A Neural Stack core state consists of hidden and cell vectors.

    Attributes:
      memory_values: Matrix V from paper.
      read_strengths: Vector s from paper.
      write_mask: Determines the index where the write will take place.
    """

    memory_values: jnp.ndarray
    read_strengths: jnp.ndarray
    write_mask: jnp.ndarray


class NeuralStackControllerInterface(NamedTuple):
    """Interface between the neural stack and it's controller.

    Attributes:
      write_values: Vector v_t from paper.
      push_strengths: Scalar d_t from paper.
      pop_strengths: Scalar u_t from paper.
    """

    write_values: jnp.ndarray
    push_strengths: jnp.ndarray
    pop_strengths: jnp.ndarray


class NeuralStackCell(hk.RNNCore):
    """
    Neural Stack from paper 'Learning to Transduce with Unbounded Memory' by
    Grefenstette et. al.. Generalized for use with stacks, queues and deques.

    Implementation with reference to:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/neural_stack.py
    https://github.com/fpjnijweide/clrs/blob/differentiable_data_structures/clrs/_src/memory_models/stack/stack.py
    """

    def __init__(
        self,
        embedding_size: int,
        memory_size: int,
        num_read_heads: int,
        num_write_heads: int,
        name: Optional[str] = "neural_stack_cell",
    ):
        """Creates a new Neural Stack Cell

        Args:
          memory_size: The maximum memory size allocated for the stack.
          embedding_size:  The embedding width of the individual stack values.
          num_read_heads: Number of read heads. 1 for stacks and queues,
                          and 2 for deque.
          num_write_heads: Number of read heads. 1 for stacks and queues,
                           and 2 for deque.
        """
        super().__init__(name=name)
        self._embedding_size = embedding_size
        self._memory_size = memory_size
        self._num_read_heads = num_read_heads
        self._num_write_heads = num_write_heads

    def initial_state(self, batch_size: Optional[int]) -> NeuralStackState:
        if batch_size is None:
            raise ValueError("Need a batch_size to get the initial state.")
        return NeuralStackState(
            memory_values=jnp.zeros([batch_size, self._memory_size, self._embedding_size]),
            read_strengths=jnp.zeros([batch_size, 1, self._memory_size, 1]),
            write_mask=self.initalize_write_mask(batch_size),
        )

    def initalize_write_mask(self, batch_size: int) -> jnp.ndarray:
        return jnp.expand_dims(
            jax.nn.one_hot(
                [[0] * self._num_write_heads] * batch_size,
                num_classes=self._memory_size,
                dtype=jnp.float32,
            ),
            axis=3,
        )

    def __call__(
        self, inputs: NeuralStackControllerInterface, prev_state: NeuralStackState
    ) -> Tuple[jnp.ndarray, NeuralStackState]:
        """Evaluates one timestep of the neural stack cell.

        Args:
          inputs: Input is in the form of an object of the interface class.
          prev_state: The previous state of the stack.

        Returns:
          Output of the single inference step in the form of a 2-tuple
          (Read value, New stack state).
        """
        # Equation-1 in Grefenstette et al., 2015.
        new_memory_values = prev_state.memory_values + jnp.sum(
            jnp.expand_dims(inputs.write_values, axis=2) * prev_state.write_mask, axis=1
        )

        # Equation-2 in Grefenstette et al., 2015.
        new_read_strengths = prev_state.read_strengths
        for h in range(self._num_read_heads - 1, -1, -1):
            new_read_strengths = jax.nn.relu(
                new_read_strengths
                - jax.nn.relu(
                    # jax.lax.slice(inputs.pop_strengths, [0, h, 0, 0], [-1, 1, -1, -1])
                    inputs.pop_strengths[:, h:h + 1, :, :] -
                    - jnp.expand_dims(
                        jnp.sum(new_read_strengths * self.get_read_mask(h), axis=2),
                        axis=3,
                    )
                )
            )
        new_read_strengths += jnp.sum(
            inputs.push_strengths * prev_state.write_mask, axis=1, keepdims=True
        )  # This is needed for deque which has different write heads (top & bottom)

        # Equation-3 in Grefenstette et al., 2015.
        new_read_values = jnp.sum(
            jnp.minimum(
                new_read_strengths,
                jax.nn.relu(
                    1
                    - jnp.expand_dims(
                        jnp.sum(
                            new_read_strengths
                            * jnp.concatenate(
                                [
                                    self.get_read_mask(h)
                                    for h in range(self._num_read_heads)
                                ],
                                axis=1,
                            ),
                            axis=2,
                        ),
                        axis=3,
                    )
                ),
            )
            * jnp.expand_dims(new_memory_values, axis=1),
            axis=2,
        )

        # Updating the write mask (first need to split for deque)
        write_masks_by_head = jnp.split(
            prev_state.write_mask, self._num_write_heads, axis=1
        )
        new_write_mask = jnp.concatenate(
            [
                jnp.roll(write_mask, shift=self.get_write_head_offset(h), axis=2)
                for h, write_mask in enumerate(write_masks_by_head)
            ],
            axis=1,
        )

        return new_read_values, NeuralStackState(
            memory_values=new_memory_values,
            read_strengths=new_read_strengths,
            write_mask=new_write_mask,
        )

    def get_read_mask(self, read_head_index: int) -> jnp.ndarray:
        """
        Creates a mask which allows us to attenuate subsequent read strengths.

        Args:
          read_head_index: Identifies which read head we're getting the mask for.

        Returns:
          An array of shape [1, 1, memory_size, memory_size]
        """
        if read_head_index == 0:
            return jnp.expand_dims(
                mask_pos_lt(self._memory_size, self._memory_size), axis=0
            )
        else:
            raise ValueError("Read head index must be 0 for stack.")

    def get_write_head_offset(self, write_head_index) -> int:
        if write_head_index == 0:
            return 1
        else:
            raise ValueError("Write head index must be 0 for stack.")


def mask_pos_lt(source_length: int, target_length: int):
    """A mask with 1.0 wherever source_pos < target_pos and 0.0 elsewhere.
    Args:
      source_length: an integer
      target_length: an integer
    Returns:
      a Tensor with shape [1, target_length, source_length]
    """
    return jnp.expand_dims(
        jax.lax.convert_element_type(
            jnp.less(
                jnp.expand_dims(jnp.arange(target_length), axis=0),
                jnp.expand_dims(jnp.arange(source_length), axis=1),
            ),
            new_dtype=jnp.float32,
        ),
        axis=0,
    )
