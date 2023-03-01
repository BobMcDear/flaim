"""
Activation functions:
- gelu: Identical to Flax's GELU but approximate is fixed at False.
- quick_gelu: Faster but more inaccurate approximation of GELU by Hendrycks et al.
- mish: Mish by Misra.
- square_relu: Squared ReLU by So et al.
- StarReLU: StarReLU by Yu et al.

References: 
- Hendrycks et al. Gaussian Error Linear Units (GELUs).
- Misra. Mish: A Self Regularized Non-Monotonic Activation Function.
- So et al. Primer: Searching for Efficient Transformers for Language Modeling.
- Yu et al. MetaFormer Baselines for Vision.
"""


__all__ = [
	'gelu',
	'quick_gelu',
	'mish',
	'squared_relu',
	'StarReLU',
	]


from functools import partial

from flax import linen as nn
from jax import numpy as jnp


gelu = partial(nn.gelu, approximate=False)


def quick_gelu(input):
	"""
	Faster but more inaccurate approximation of GELU.

	Args:
		input: Input.
	
	Returns: Result of quick GELU.
	"""
	return input * nn.sigmoid(1.702*input)


def mish(input):
	"""
	Mish activation function.

	Args:
		input: Input.
	
	Returns: Result of mish.
	"""
	return input * jnp.tanh(nn.softplus(input))


def squared_relu(input):
	"""
	Squared ReLU activation function.

	Args:
		input: Input.
	
	Returns: Result of squared ReLU.
	"""
	return nn.relu(input) ** 2


class StarReLU(nn.Module):
	"""
	StarReLU activation function.

	Args:
		scale_init_value (float): Initial value for the
		scale parameter.
		Default is 1.
		bias_init_value (float): Initial value for the
		bias parameter.
		Default is 0.
	"""
	scale_init_value: float = 1.0
	bias_init_value: float = 0.0
	
	@nn.compact
	def __call__(self, input):
		scale = self.param(
			name='scale',
			init_fn=lambda prng: self.scale_init_value,
			)
		bias = self.param(
			name='bias',
			init_fn=lambda prng: self.bias_init_value,
			)
		return scale * squared_relu(input) + bias
