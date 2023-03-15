"""
Convolution similar to that of Flax but can automatically calulate
padding, accepts integer kernel sizes, supports weight standardization, etc.
"""


__all__ = [
	'Conv',
	]


import typing as T
from functools import partial

import jax
from flax import linen as nn

from .tuplify import tuplify


def get_kernel_size_stride_padding(
	kernel_size: T.Union[T.Tuple[int, int], int] = 3,
	stride: T.Union[T.Tuple[int, int], int] = 1,
	padding: T.Optional[T.Union[str, int]] = None,
	dilation: int = 1,
	) -> T.Tuple[T.Tuple[int, int], T.Tuple[int, int], T.Tuple[T.Tuple[int, int], T.Tuple[int, int]]]:
	"""
	Gets the appropriate kernel size, stride, and padding to be passed to a convolution.

	Args:
		kernel_size (T.Union[T.Tuple[int, int], int]): Kernel size.
		If an int, this value is used along both spatial dimensions.
		Default is 3.
		stride (T.Union[T.Tuple[int, int], int]): Stride. If an int,
		this value is used along both spatial dimensions.
		Default is 1.
		padding (T.Optional[T.Union[str, int]]): Padding. If None,
		it is set so the spatial dimensions are exactly divided by stride.
		If an int, this value is used along both spatial dimensions.
		Default is None.
		dilation (int): Dilation.
		Default is 1.
	
	Returns (T.Tuple[T.Tuple[int, int], T.Tuple[int, int], T.Tuple[T.Tuple[int, int], T.Tuple[int, int]]]):
	Kernel size, stride, and padding.
	"""
	kernel_size = tuplify(kernel_size)
	stride = tuplify(stride)

	if isinstance(padding, int):
		padding = 2*(tuplify(padding),)

	elif padding is None:
		padding = (tuplify((dilation * (kernel_size[0]-1)) // 2), tuplify((dilation * (kernel_size[1]-1)) // 2))
	
	return kernel_size, stride, padding


class WSConv(nn.Conv):
	"""
	Convolution with weight standardization.
	
	Args:
		eps (float): Epsilon value used in the denominator 
		when standardizing.
		Default is 1e-5.

	Please also see flax.linen.Conv
	"""
	eps: float = 1e-5

	def param(
		self,
		name: str,
		*args,
		**kwargs,
		):
		params = super().param(name, *args, **kwargs)
		if name == 'kernel':
			params = jax.nn.standardize(params, axis=(0, 1, 2), epsilon=self.eps)
		return params


class Conv(nn.Module):
	"""
	Convolution similar to that of Flax but can automatically calulate
	padding, accepts integer kernel sizes, supports weight standardization, etc.

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
		padding (T.Optional[T.Union[str, int]]): Padding. If None,
		it is set so the spatial dimensions are exactly divided by stride.
		If an int, this value is used along both spatial dimensions.
		Default is None.
		groups (T.Union[int, str]): Number of groups. If 'dw', a depthwise convolution is performed.
		Default is 1.
		dilation (int): Dilation.
		Default is 1.
		bias (bool): Whether to have a bias term.
		Default is True.
		ws_eps (T.Optional[float]): Epsilon value for weight standardization. If None,
		no weight standardization is performed.
		Default is None.
	"""
	out_dim: T.Optional[int] = None
	kernel_size: T.Union[T.Tuple[int, int], int] = 3
	stride: T.Union[T.Tuple[int, int], int] = 1
	padding: T.Optional[T.Union[str, int]] = None
	groups: T.Union[int, str] = 1
	dilation: int = 1
	bias: bool = True
	ws_eps: T.Optional[float] = None

	@nn.compact
	def __call__(self, input):
		in_dim = input.shape[-1]
		out_dim = self.out_dim or in_dim
		kernel_size, stride, padding = get_kernel_size_stride_padding(
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding,
			dilation=self.dilation,
			)
		conv = partial(WSConv, eps=self.ws_eps) if self.ws_eps else nn.Conv
		return conv(
			features=out_dim,
			kernel_size=kernel_size,
			strides=stride,
			padding=padding,
			feature_group_count=in_dim if self.groups == 'dw' else self.groups,
			kernel_dilation=self.dilation,
			use_bias=self.bias,
			)(input)
