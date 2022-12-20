"""
Functions related to shifted window attention by Liu et al.:
- cyclic_shift: Cyclical shift.
- window_partition: Partitions the input into windows.
- window_merge: Merges windows.

References:
- Liu et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
"""


__all__ = [
	'cyclic_shift',
	'window_partition',
	'window_merge',
	]


import typing as T

from jax import numpy as jnp

from .tuplify import tuplify


def cyclic_shift(
	input,
	shift_size: T.Union[T.Tuple[int, int], int],
	):
	"""
	Cyclic-shifts the input.

	Args:
		input: Input.
		shift_size (T.Union[T.Tuple[int, int], int]): Shift size.
		If an int, this value is used along both spatial dimensions.
	
	Returns: Cyclic-shifted version of input.
	"""
	shift_size = tuplify(shift_size)
	return jnp.roll(
		input,
		shift=(-shift_size[0], -shift_size[1]),
		axis=(-3, -2),
		)


def window_partition(
	input,
	window_size: T.Union[T.Tuple[int, int], int],
	):
	"""
	Partitions the input into windows.

	Args:
		input: Input.
		window_size (T.Union[T.Tuple[int, int], int]): Window size.
		If an int, this value is used along both spatial dimensions.
	
	Returns: The input partitioned into windows.
	"""
	bs, h, w, in_dim = input.shape
	window_h, window_w = tuplify(window_size)

	output = jnp.reshape(input, (bs, h//window_h, window_h, w//window_w, window_w, in_dim))
	output = jnp.transpose(output, (0, 1, 3, 2, 4, 5))
	output = jnp.reshape(output, (-1, window_h, window_w, in_dim))
	return output


def window_merge(
	input,
	img_size: T.Union[T.Tuple[int, int], int],
	window_size: T.Union[T.Tuple[int, int], int],
	):
	"""
	Merges windows.

	Args:
		input: Input.
		img_size (T.Union[T.Tuple[int, int], int]): Image size.
		If an int, this value is used along both spatial dimensions.
		window_size (T.Union[T.Tuple[int, int], int]): Window size.
		If an int, this value is used along both spatial dimensions.
	
	Returns: The merged version of the input's windows.
	"""
	c = input.shape[-1]
	img_h, img_w = tuplify(img_size)
	window_h, window_w = tuplify(window_size)

	output = jnp.reshape(input, (-1, img_h//window_h, img_w//window_w, window_h, window_w, c))
	output = jnp.transpose(output, (0, 1, 3, 2, 4, 5))
	output = jnp.reshape(output, (-1, img_h, img_w, c))
	return output
