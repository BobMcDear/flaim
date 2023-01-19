"""
Classification heads:
- Head: Standard head with pooling, layer normalization, and a linear layer.
- ViTHead: Head for ViTs with optional pooling of tokens, layer normalization,
and a linear layer.
"""


__all__ = [
	'Head',
	'ViTHead'
	]


import typing as T

from flax import linen as nn
from jax import numpy as jnp

from .pool import global_avg_pool


class Head(nn.Module):
	"""
	Standard head with pooling, layer normalization, and a linear layer.

	Args:
		n_classes (int): Number of output classes. If 0, the input is returned.
		If -1, all stages of the head, other than the final linear layer,
		are applied and the output returned. 
		Default is 0.
		pool_fn (T.Callable): Pooling function.
		Default is global_avg_pool.
		layer_norm_eps (T.Optional[float]): Epsilon value
		passed to layer normalization. If None, no normalization is applied,
		and norm_first is ignored.
		Default is None.
		norm_first (bool): Whether to apply layer normalization before
		pooling instead of after.
		Default is False.
		bias (bool): Whether the linear layer should have
		a bias term.
		Default is True.
	"""
	n_classes: int = 0
	pool_fn: T.Callable = global_avg_pool
	layer_norm_eps: T.Optional[float] = None
	norm_first: bool = False
	bias: bool = True

	@nn.compact
	def __call__(self, input):
		if self.n_classes == 0:
			return input

		if self.norm_first:
			output = nn.LayerNorm(
				epsilon=self.layer_norm_eps,
				)(input) if self.layer_norm_eps else input
			output = self.pool_fn(output, keep_axis=False)
		
		else:
			output = self.pool_fn(input, keep_axis=False)
			output = nn.LayerNorm(
				epsilon=self.layer_norm_eps,
				)(output) if self.layer_norm_eps else output
		
		if self.n_classes != -1:
			output = nn.Dense(
				features=self.n_classes,
				use_bias=self.bias,
				)(output)

		return output


class ViTHead(nn.Module):
	"""
	Head for ViTs with optional pooling of tokens, layer
	normalization, and a linear layer.

	Args:
		n_classes (int): Number of output classes. If 0, the input is returned.
		If -1, all stages of the head, other than the final linear layer,
		are applied and the output returned. 
		Default is 0.
		pool (bool): Whether to average pool the tokens for
		generating predictions. If False, the first token
		in the input, assumed to be the class token, is used
		to generate predictions.
		layer_norm_eps (T.Optional[float]): Epsilon value
		passed to layer normalization applied at the beginning.
		If None, no normalization is applied.
		Default is None.
		norm_first (bool): Whether to apply layer normalization before
		pooling/class token extraction instead of after.
		Default is True.
		bias (bool): Whether the linear layer should have
		a bias term.
		Default is True.
	"""
	n_classes: int = 0
	pool: bool = False
	layer_norm_eps: T.Optional[float] = None
	norm_first: bool = True
	bias: bool = True

	@nn.compact
	def __call__(self, input):
		if self.n_classes == 0:
			return input

		if self.norm_first:
			output = nn.LayerNorm(
				epsilon=self.layer_norm_eps,
				)(input) if self.layer_norm_eps else input
			output = jnp.mean(output, axis=-2) if self.pool else output[:, 0]
	
		else:
			output = jnp.mean(input, axis=-2) if self.pool else input[:, 0]
			output = nn.LayerNorm(
				epsilon=self.layer_norm_eps,
				)(output) if self.layer_norm_eps else output

		if self.n_classes != -1:
			output = nn.Dense(
				features=self.n_classes,
				use_bias=self.bias,
				)(output)

		return output
