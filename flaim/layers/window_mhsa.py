"""
Plain window multi-headed self-attention by Liu et al. without cyclical shifts,
relative position embedding, etc.

References:
- Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
"""


__all__ = [
	'WindowMHSA',
	]


from flax import linen as nn
from jax import numpy as jnp

from .mhsa import MHSA
from .window import window_merge, window_partition


class WindowMHSA(MHSA):
	"""
	Plain window multi-headed self-attention.

	Args:
		window_size (int): Window size.
		Default is 7.
	"""
	window_size: int = 7

	@nn.compact
	def __call__(self, input):
		bs, h, w, in_dim = input.shape

		output = window_partition(input, self.window_size)
		output = jnp.reshape(output, (-1, self.window_size ** 2, in_dim))

		output = super().__call__(output)

		output = window_merge(
			output,
			window_size=self.window_size,
			img_size=(h, w),
			)

		return output
