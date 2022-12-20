"""
Selective kernel unit (SK) by Li et al., with support for splitting
tweak by Wightman.

References:
- Li et al. Selective Kernel Networks.
- Wightman. PyTorch Image Models.
"""


__all__ = [
	'SK',
	]


import typing as T

from flax import linen as nn
from jax import numpy as jnp

from .lin_norm_act import ConvBNAct
from .mlp import MLP
from .pool import global_avg_pool


class SK(nn.Module):
	"""
	Selective kernel unit.

	Args:
		out_dim (T.Optional[int]): Number of output channels.
		If None, it is set to the number of input channels.
		Default is None.
		stride (int): Stride.
		groups (T.Union[int, str]): Number of groups. If 'dw', a depthwise convolution is performed.
		Default is 1.
		act (T.Callable): Activation function.
		Default is nn.relu.
		n_branches (int): Number of branches.
		Default is 2.
		reduction_factor (int): Reduction factor for the bottleneck
		MLP.
		Default is 8.
		min_reduction_dim (int): Minimum number of channels for the
		hidden layer of the bottleneck MLP.
		Default is 32.
		split (int): Whether to feed each branch a split
		of the input instead of the entire input.
		Default is True.
	"""
	out_dim: T.Optional[int] = None
	stride: int = 1
	groups: T.Union[int, str] = 1
	act: T.Callable = nn.relu
	n_branches: int = 2
	reduction_factor: int = 8
	min_reduction_dim: int = 16
	split: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		if self.split:
			splits = jnp.split(
				ary=input,
				indices_or_sections=self.n_branches,
				axis=-1,
				)
		
		bs, h, w, in_dim = input.shape
		out_dim = self.out_dim or in_dim
		output = []
		for branch_ind in range(self.n_branches):
			output.append(
				ConvBNAct(
					out_dim=out_dim,
					stride=self.stride,
					groups=self.groups,
					dilation=branch_ind+1,
					act=self.act,
					)(splits[branch_ind] if self.split else input, training=training),
				)
		output = jnp.stack(output, axis=-2)

		attention = jnp.sum(output, axis=-2)
		attention = global_avg_pool(attention)
		attention = MLP(
			out_dim=self.n_branches*out_dim,
			hidden_dim=max(out_dim//self.reduction_factor, self.min_reduction_dim),
			act=self.act,
			bias=False,
			bn=True,
			)(attention, training=training)
		attention = jnp.reshape(attention, (bs, 1, 1, self.n_branches, out_dim))
		attention = nn.softmax(attention, axis=-2)

		output = attention*output
		output = jnp.sum(output, axis=-2)
		return output
