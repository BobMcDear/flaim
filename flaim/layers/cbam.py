"""
Convolutional block attention module (CBAM) by Woo et al.

References:
- Woo et al. CBAM: Convolutional Block Attention Module.
"""


__all__ = [
	'CBAM',
	]


import typing as T

from flax import linen as nn

from .conv import Conv
from .mlp import MLP
from .pool import global_avg_max_pool, global_concat_avg_max_pool


class CBAMChannelAttention(nn.Module):
	"""
	CBAM's channel attention module.

	Args:
		reduction_factor (T.Optional[int]): Factor of reduction for the 
		hidden layer of MLP. If None, reduction_dim must be provided.
		If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is 16.
		reduction_dim (T.Optional[int]): Number of channels in the 
		hidden layer of the MLP. If None, reduction_factor must be provided.
		If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is None.
		act (T.Callable): Activation function.
		Default is nn.relu.
		gate (T.Callable): Gating function used to normalize the attention
		scores.
	"""
	reduction_factor: T.Optional[int] = 16
	reduction_dim: T.Optional[int] = None
	act: T.Callable = nn.relu
	gate: T.Callable = nn.sigmoid

	@nn.compact
	def __call__(self, input):
		mlp = MLP(
			hidden_dim_expansion_factor=1/self.reduction_factor if self.reduction_factor else None,
			hidden_dim=self.reduction_dim,
			act=self.act,
			)

		avg_attention, max_attention = global_avg_max_pool(input)
		avg_attention = mlp(avg_attention)
		max_attention = mlp(max_attention)
		
		attention = self.gate(avg_attention + max_attention)
		return attention*input


class CBAMSpatialAttention(nn.Module):
	"""
	CBAM's spatial attention module.

	Args:
		gate (T.Callable): Gating function used to normalize the attention
		scores.
	"""
	kernel_size: int = 7
	gate: T.Callable = nn.sigmoid

	@nn.compact
	def __call__(self, input):
		attention = global_concat_avg_max_pool(input, axis=-1)
		attention = Conv(
			out_dim=1,
			kernel_size=self.kernel_size,
			)(attention)
		return attention*input


class CBAM(nn.Module):
	"""
	Convolutional block attention module.

	Args:
		reduction_factor (T.Optional[int]): Factor of reduction for the 
		hidden layer of MLP of the spatial attention module. If None,
		reduction_dim must be provided. If both reduction_factor and
		reduction_dim are provided, the latter takes precedence.
		Default is 16.
		reduction_dim (T.Optional[int]): Number of channels in the 
		hidden layer of the MLP of the channel attention module.
		If None, reduction_factor must be provided. If both reduction_factor
		and reduction_dim are provided, the latter takes precedence.
		Default is None.
		kernel_size (int): Kernel size for the spatial attention module.
		Default is 7.
		act (T.Callable): Activation for the excitation module.
		Default is nn.relu.
		gate (T.Callable): Gating function used to normalize the attention
		scores.
	"""
	reduction_factor: T.Optional[int] = 16
	reduction_dim: T.Optional[int] = None
	kernel_size: int = 7
	act: T.Callable = nn.relu
	gate: T.Callable = nn.sigmoid

	@nn.compact
	def __call__(self, input):
		output = CBAMChannelAttention(
			reduction_factor=self.reduction_factor,
			reduction_dim=self.reduction_dim,
			act=self.act,
			gate=self.gate,
			)(input)
		output = CBAMSpatialAttention(
			kernel_size=self.kernel_size,
			gate=self.gate,
			)(output)
		return output
