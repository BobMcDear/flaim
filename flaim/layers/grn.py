"""
Global response normalization (GRN) by Woo et al.

References:
- Woo et al. ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.
"""


__all__ = [
	'GRN',
	]


from flax import linen as nn
from jax import numpy as jnp


class GRN(nn.Module):
	"""
	Global response normalization.

	Args:
		eps (float): Epsilon value for the denominator.
		Default is 1e-6.
	"""
	eps: float = 1e-6

	@nn.compact
	def __call__(self, input):
		in_dim = input.shape[-1]
		gamma = self.param(
			name='gamma',
			init_fn=lambda prng: jnp.zeros(in_dim),
			)
		beta = self.param(
			name='beta',
			init_fn=lambda prng: jnp.zeros(in_dim),
			)

		gx = jnp.linalg.norm(input, axis=(1, 2), keepdims=True)
		nx = gx / (jnp.mean(gx, axis=-1, keepdims=True) + self.eps)
		
		return input + gamma * (input * nx) + beta
