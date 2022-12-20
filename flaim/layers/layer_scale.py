"""
LayerScale by Touvron et al.

References:
- Touvron et al. Going deeper with Image Transformers.
"""


__all__ = [
	'LayerScale',
	]


import typing as T

from flax import linen as nn
from jax import numpy as jnp


class LayerScale(nn.Module):
	"""
	LayerScale.

	Args:
		init_value (T.Optional[float]): Initial value for the
		lambda parameter. If None, no LayerScale is applied.
		Default is 1e-6.
	"""
	init_value: T.Optional[float] = 1e-6

	@nn.compact
	def __call__(self, input):
		return input * self.param(
			name='lambda',
			init_fn=lambda prng: self.init_value*jnp.ones(input.shape[-1]),
			) if self.init_value else input
