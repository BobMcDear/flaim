"""
Efficient channel attention (ECA) by Wang et al.

References:
- Wang et al. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks.
"""


__all__ = [
	'ECA',
	]


import typing as T
from math import log

from flax import linen as nn
from jax import numpy as jnp

from .pool import global_avg_pool


def get_kernel_size(
	in_dim: int,
	beta: int = 1,
	gamma: int = 2,
	) -> int:
	"""
	Calculates the optimal kernel size for ECA.

	Args:
		in_dim (int): Number of input channels.
		beta (int): Beta parameter.
		Default is 1.
		gamma (int): Gamma parameter.
		Default is 2.
	
	Returns (int): Optimal kernel size for ECA given in_dim.
	"""
	t = int((log(in_dim, 2) + beta) / gamma)
	kernel_size = t if t%2 == 1 else t+1
	return kernel_size


class ECA(nn.Module):
	"""
	Efficient channel attention.

	Args:
		beta (int): Beta parameter for calculating the kernel
		size.
		Default is 1.
		gamma (int): Gamma parameter for calculating the kernel
		size.
		Default is 2.
		kernel_size (T.Optional[int]): Kernel size. If None, beta
		and gamma are used to calculate the kernel size. If passed,
		beta and gamma are ignored.
		Default is None.
		pool (T.Callable): Pooling method used for aggregating the
		activations of each channel.
		Default is global_avg_pool.
		gate (T.Callable): Gating function used to normalize the
		attention scores.
		Default is nn.sigmoid.
	"""
	beta: int = 1
	gamma: int = 2
	kernel_size: T.Optional[int] = None
	pool: T.Callable = global_avg_pool
	gate: T.Callable = nn.sigmoid

	@nn.compact
	def __call__(self, input):
		attention = self.pool(input, keep_axis=False)
		attention = jnp.expand_dims(attention, axis=-2)
		attention = jnp.swapaxes(attention, axis1=-2, axis2=-1)
		
		kernel_size = self.kernel_size or get_kernel_size(input.shape[-1])
		attention = nn.Conv(
			features=1,
			kernel_size=(kernel_size,),
			padding=(kernel_size//2,),
			use_bias=False,
			)(attention)
		attention = self.gate(attention)

		attention = jnp.swapaxes(attention, axis1=-2, axis2=-1)
		attention = jnp.expand_dims(attention, axis=-2)

		return attention*input
