"""
Split attention (SplAt) by Zhang et al.

References:
- Zhang et al. ResNeSt: Split-Attention Networks.
"""


__all__ = [
	'SplAt',
	]


import typing as T

from flax import linen as nn
from jax import numpy as jnp

from .lin_norm_act import ConvBNAct
from .mlp import ConvMLP
from .pool import global_avg_pool


class RadixSoftmax(nn.Module):
	"""
	Softmax along the radix axis.

	Args:
		radix (int): Radix. When radix is 1, sigmoid
		is applied.
		Default is 2.
		groups (int): Number of cardinal groups.
		Default is 1.
	"""
	radix: int = 2
	groups: int = 1

	def __call__(self, input):
		bs, h, w, in_dim = input.shape

		if 1 < self.radix:
			output = jnp.reshape(input, (bs, h, w, self.groups, self.radix, -1))
			output = jnp.swapaxes(output, axis1=-3, axis2=-2)
			output = nn.softmax(output, axis=-3)
			output = jnp.reshape(output, (bs, h, w, self.radix, -1))

		else:
			output = nn.sigmoid(input)

		return output


class SplAt(nn.Module):
	"""
	Split attention.

	Args:
		out_dim (T.Optional[int]): Number of output channels.
		If None, it is set to the number of input channels.
		Default is None.
		kernel_size (T.Union[T.Tuple[int, int], int]): Kernel size.
		If an int, this value is used along both spatial dimensions.
		Default is 3.
		stride (T.Union[T.Tuple[int, int], int]): Stride. If an int,
		this value is used along both spatial dimensions.
		Default is 1.
		groups (int): Number of cardinal groups.
		Default is 1.
		act (T.Callable): Activation function.
		Default is nn.relu.
		radix (int): Radix.
		Default is 2.
		reduction_factor (int): Reduction factor for the bottleneck
		MLP.
		Default is 4.
		min_reduction_dim (int): Minimum number of channels for the
		hidden layer of the bottleneck MLP.
		Default is 32.
	"""
	out_dim: T.Optional[int] = None
	kernel_size: T.Union[T.Tuple[int, int], int] = 3
	stride: T.Union[T.Tuple[int, int], int] = 1
	groups: int = 1
	act: T.Callable = nn.relu
	radix: int = 2
	reduction_factor: int = 4
	min_reduction_dim: int = 32

	@nn.compact
	def __call__(self, input, training: bool = True):
		bs, h, w, in_dim = input.shape
		out_dim = self.out_dim or in_dim
		width = self.radix*out_dim

		output = ConvBNAct(
			out_dim=width,
			kernel_size=self.kernel_size,
			stride=self.stride,
			groups=self.radix*self.groups,
			act=self.act,
			)(input, training=training)

		if 1 < self.radix:
			output = jnp.reshape(output, (bs, h, w, self.radix, out_dim))
			attention = jnp.sum(output, axis=-2)

		else:
			attention = output

		attention = global_avg_pool(attention)
		attention = ConvMLP(
			out_dim=width,
			hidden_dim=max(width//self.reduction_factor, self.min_reduction_dim),
			groups=self.groups,
			act=self.act,
			bias_force=True,
			bn=True,
			)(attention, training=training)
		attention = RadixSoftmax(
			radix=self.radix,
			groups=self.groups,
			)(attention)

		output = attention*output
		if 1 < self.radix:
			output = jnp.sum(output, axis=-2)

		return output
