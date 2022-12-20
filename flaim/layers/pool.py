"""
Pooling methods:
- avg_pool: Pooling similar to Flax's average pooling but can automatically calulate
padding, accepts integer kernel size, etc.
- max_pool: Pooling similar to Flax's average pooling but can automatically calulate
padding, accepts integer kernel size, etc.
- global_avg_pool: Global average pooling.
- global_max_pool: Global max pooling.
- global_avg_max_pool: Returns a tuple of the globally average- and max-pooled input.
- global_concat_avg_max_pool: Concatenates the results of global 
average and max pooling, by Wightman.
- global_sum_avg_max_pool: Sums the results of global 
average and max pooling, by Wightman.

References:
- Wightman. PyTorch Image Models.
"""


__all__ = [
	'avg_pool',
	'max_pool',
	'global_avg_pool',
	'global_max_pool',
	'global_avg_max_pool',
	'global_concat_avg_max_pool',
	'global_sum_avg_max_pool',
	]


import typing as T

from flax import linen as nn
from jax import numpy as jnp

from .conv import get_kernel_size_stride_padding


def avg_pool(
	input,
	kernel_size: T.Union[T.Tuple[int, int], int] = 3,
	stride: T.Union[T.Tuple[int, int], int] = 1,
	padding: T.Optional[T.Union[str, int]] = None,
	count_include_pad: bool = True,
	):
	"""
	Similar to Flax's average pooling but accepts integer kernel size, 
	supports depthwise convolution, etc.

	Args:
		input: Input.
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
		count_include_pad (bool): Whether to include padding when
		calculating averages.
		Default is True.
	"""
	kernel_size, stride, padding = get_kernel_size_stride_padding(
		kernel_size=kernel_size,
		stride=stride,
		padding=padding,
		)
	return nn.avg_pool(
		inputs=input,
		window_shape=kernel_size,
		strides=stride,
		padding=padding,
		count_include_pad=count_include_pad,
		)


def max_pool(
	input,
	kernel_size: T.Union[T.Tuple[int, int], int] = 3,
	stride: T.Union[T.Tuple[int, int], int] = 1,
	padding: T.Optional[T.Union[str, int]] = None,
	):
	"""
	Similar to Flax's max pooling but accepts integer kernel size, 
	supports depthwise convolution, etc.

	Args:
		input: Input.
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
	"""
	kernel_size, stride, padding = get_kernel_size_stride_padding(
		kernel_size=kernel_size,
		stride=stride,
		padding=padding,
		)
	return nn.max_pool(
		inputs=input,
		window_shape=kernel_size,
		strides=stride,
		padding=padding,
		)


def global_avg_pool(
	input,
	retain_spatial: bool = True,
	):
	"""
	Global  average pooling.

	Args:
		input: Input.
		retain_spatial (bool): Whether the spatial
		dimensions should be retained or squeezed.
		Default is True.
	
	Returns: Globally average-pooled input.
	"""
	return jnp.mean(input, axis=(-3, -2), keepdims=retain_spatial)


def global_max_pool(
	input,
	retain_spatial: bool = True,
	):
	"""
	Global  max pooling.

	Args:
		input: Input.
		retain_spatial (bool): Whether the spatial
		dimensions should be retained or squeezed.
		Default is True.
	
	Returns: Globally average-pooled input.
	"""
	return jnp.max(input, axis=(-3, -2), keepdims=retain_spatial)


def global_avg_max_pool(
	input,
	retain_spatial: bool = True,
	):
	"""
	Returns a tuple of the globally average- and max-pooled input.

	Args:
		input: Input.
		retain_spatial (bool): Whether the spatial
		dimensions should be retained or squeezed.
		Default is True.
	
	Returns: Tuple of the globally average- and max-pooled input.
	"""
	return (
		global_avg_pool(input, retain_spatial),
		global_max_pool(input, retain_spatial),
		)


def global_concat_avg_max_pool(
	input,
	retain_spatial: bool = True,
	):
	"""
	Concatenates the results of global  average and max pooling.

	Args:
		retain_spatial (bool): Whether the spatial
		dimensions should be retained or squeezed.
		Default is True.
	
	Returns: Concatenation of the globally average- and max-pooled input.
	"""
	return jnp.concatenate(global_avg_max_pool(input, retain_spatial), axis=-1)


def global_sum_avg_max_pool(
	input,
	retain_spatial: bool = True,
	):
	"""
	Sums the results of global  average and max pooling.

	Args:
		retain_spatial (bool): Whether the spatial
		dimensions should be retained or squeezed.
		Default is True.
	
	Returns: Summation of the globally average- and max-pooled input.
	"""
	return jnp.sum(global_avg_max_pool(input, retain_spatial), axis=-1)
