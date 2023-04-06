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

from .identity import identity
from .mlp import MLP
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
		hidden_dim (T.Optional[int]): If not None, the logits are transformed
		to this dimension using a linear layer followed by an activation function
		before being passed to the final linear layer, that is, the head turns into an
		MLP. hidden_act is ignored if hidden_dim is None.
		Default is None.
		hidden_act (T.Callable): Activation function used if hidden_dim is
		not None.
		Default is nn.tanh.
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
	hidden_dim: T.Optional[int] = None
	hidden_act: T.Callable = nn.tanh
	layer_norm_eps: T.Optional[float] = None
	norm_first: bool = False
	bias: bool = True

	@nn.compact
	def __call__(self, input):
		if self.n_classes == 0:
			return input

		layer_norm = nn.LayerNorm(
			epsilon=self.layer_norm_eps,
			) if self.layer_norm_eps else identity

		if self.norm_first:
			output = layer_norm(input)
			output = self.pool_fn(output, keep_axis=False)

		else:
			output = self.pool_fn(input, keep_axis=False)
			output = layer_norm(output)

		if self.hidden_dim:
			output = nn.Sequential([
				nn.Dense(features=self.hidden_dim),
				self.hidden_act,
				])(output)

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
		hidden_dim (T.Optional[int]): If not None, the logits are transformed
		to this dimension using a linear layer followed by an activation function
		before being passed to the final linear layer, that is, the head turns into an
		MLP. hidden_act is ignored if hidden_dim is None.
		Default is None.
		hidden_act (T.Callable): Activation function used if hidden_dim is
		not None.
		Default is nn.tanh.
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
	hidden_dim: T.Optional[int] = None
	hidden_act: T.Callable = nn.tanh
	layer_norm_eps: T.Optional[float] = None
	norm_first: bool = True
	bias: bool = True

	@nn.compact
	def __call__(self, input):
		if self.n_classes == 0:
			return input

		layer_norm = nn.LayerNorm(
			epsilon=self.layer_norm_eps,
			) if self.layer_norm_eps else identity

		if self.norm_first:
			output = layer_norm(input)
			output = jnp.mean(output, axis=-2) if self.pool else output[:, 0]

		else:
			output = jnp.mean(input, axis=-2) if self.pool else input[:, 0]
			output = layer_norm(output)

		if self.hidden_dim:
			output = nn.Sequential([
				nn.Dense(features=self.hidden_dim),
				self.hidden_act,
				])(output)

		if self.n_classes != -1:
			output = nn.Dense(
				features=self.n_classes,
				use_bias=self.bias,
				)(output)

		return output
