"""
Bottleneck attention module (BAM) by Park et al.

References:
- Park et al. BAM: Bottleneck Attention Module.
"""


__all__ = [
	'BAM',
	]


import typing as T

from flax import linen as nn

from .conv import Conv
from .lin_norm_act import ConvBNAct
from .pool import global_avg_pool
from .se import SE


class BAMSpatialAttention(nn.Module):
	"""
	BAM's spatial attention module.

	Args:
		reduction_factor (T.Optional[int]): Factor of reduction for the 
		3 x 3 convolutions. If None, reduction_dim must be provided.
		If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is 16.
		reduction_dim (T.Optional[int]): Number of channels for the
		3 x 3 convolutions. If None, reduction_factor must be provided.
		If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is None.
		dilation (int): Dilation of the 3 x 3 convolutions.
		Default is 4.
		act (T.Callable): Activation function.
		Default is nn.relu.
	"""
	reduction_factor: T.Optional[int] = 16
	reduction_dim: T.Optional[int] = None
	dilation: int = 4
	act: T.Callable = nn.relu

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = ConvBNAct(
			out_dim=self.reduction_dim or (input.shape[-1] // self.reduction_factor),
			kernel_size=1,
			act=self.act,
			)(input, training=training)
		
		output = ConvBNAct(
			dilation=self.dilation,
			act=self.act,
			)(output, training=training)
		output = ConvBNAct(
			dilation=self.dilation,
			act=self.act,
			)(output, training=training)
		
		output = Conv(
			out_dim=1,
			kernel_size=1,
			)(output)
		
		return output


class BAM(nn.Module):
	"""
	Bottleneck attention module.

	Args:
		reduction_factor (T.Optional[int]): Factor of reduction for the 
		bottlenecks. If None, reduction_dim must be provided.
		If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is 16.
		reduction_dim (T.Optional[int]): Number of channels for the
		bottlenecks. If None, reduction_factor must be provided.
		If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is None.
		dilation (int): Dilation for the spatial attention module.
		Default is 4.
		pool (T.Callable): Pooling method for the channel attention module.
		Default is global_avg_pool.
		act (T.Callable): Activation function.
		Default is nn.relu.
		gate (T.Callable): Gating function used to normalize the attention
		scores.
		Default is nn.sigmoid.
	"""
	reduction_factor: T.Optional[int] = 16
	reduction_dim: T.Optional[int] = None
	dilation: int = 4
	pool: T.Callable = global_avg_pool
	act: T.Callable = nn.relu
	gate: T.Callable = nn.sigmoid

	@nn.compact
	def __call__(self, input, training: bool = True):
		channel_attn = SE(
			reduction_factor=self.reduction_factor,
			reduction_dim=self.reduction_dim,
			pool=self.pool,
			act=self.act,
			bn=True,
			return_attn=True,
			)(input, training=training)
		spatial_attn = BAMSpatialAttention(
			reduction_factor=self.reduction_factor,
			reduction_dim=self.reduction_dim,
			dilation=self.dilation,
			act=self.act,
			)(input, training=training)
			
		attention = self.gate(channel_attn * spatial_attn) + 1
		return attention * input
