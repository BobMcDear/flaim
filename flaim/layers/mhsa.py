"""
Multi-headed self-attention (MHSA) and some related modules by Vaswani et al.:
- QKV: Query-key-value extractor.
- scaled_dot_product_similarity: Scaled dot-product similarity function.
- ProjOut: MHSA's regular output projection mechanism.
- MHSA: Multi-headed self-attention.

References:
- Vaswani et al. Attention Is All You Need.
"""


__all__ = [
	'QKV',
	'scaled_dot_product_similarity',
	'ProjOut',
	'MHSA',
	]


import typing as T

from flax import linen as nn
from jax import numpy as jnp

from .identity import Identity


class QKV(nn.Module):
	"""
	Exctracts query, key, and value vectors for multi-headed self-attention.

	Args:
		n_heads (int): Number of heads.
		bias (bool): Whether the linear transformations should have
		a bias term. If False, k_bias is ignored.
		Default is True.
		k_bias (bool): Whether the linear transformation
		for obtaining keys should contain a bias term.
		Default is True.
	"""
	n_heads: int
	bias: bool = True
	k_bias: bool = True

	@nn.compact
	def __call__(self, input):
		n_tokens, token_dim = input.shape[-2:]
		head_dim = token_dim//self.n_heads

		if not self.bias:
			qkv = nn.Dense(
				features=3*token_dim,
				use_bias=False,
				)(input)

		elif self.k_bias:
			qkv = nn.Dense(
				features=3*token_dim,
				)(input)

		else:
			q_bias = self.param('q_bias', lambda prng: jnp.zeros((token_dim)))
			k_bias = self.variable('k_bias_', 'k_bias_', lambda: jnp.zeros((token_dim)))
			v_bias = self.param('v_bias', lambda prng: jnp.zeros((token_dim)))
			qkv = nn.Dense(
				features=3*token_dim,
				use_bias=False,
				)(input) + jnp.concatenate([q_bias, k_bias.value, v_bias])

		qkv = jnp.reshape(qkv, (-1, n_tokens, 3, self.n_heads, head_dim))
		qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
		q, k, v = jnp.split(
			ary=qkv,
			indices_or_sections=3,
			axis=0,
			)
			
		return jnp.squeeze(q, axis=0), jnp.squeeze(k, axis=0), jnp.squeeze(v, axis=0)


def scaled_dot_product_similarity(q, k):
	"""
	Scaled dot-product similarity.

	Args:
		q: Queries.
		k: Keys.
	
	Returns: Scaled dot-product similarity between the queries and keys.
	"""
	return q @ jnp.swapaxes(k, axis1=-2, axis2=-1) / jnp.sqrt(q.shape[-1])


class ProjOut(nn.Module):
	"""
	MHSA's regular output projection mechanism.
	"""
	@nn.compact
	def __call__(self, input):
		output = jnp.swapaxes(input, axis1=-2, axis2=-3)
		output = jnp.reshape(output, (len(input), input.shape[-2], -1))
		output = nn.Dense(
			features=output.shape[-1],
			)(output)
		return output


class MHSA(nn.Module):
	"""
	Multi-headed self-attention module.

	Args:
		to_qkv (T.Optional[T.Union[T.Callable, int]]): If input is not None,
		to_qkv should return the module used to extract queries, key, and values.
		If an int, the regular QKV extractor is used, and to_qkv is interpreted 
		as the number of heads. If input is None, to_qkv is ignored.
		Default is None.
		similarity_fn (T.Callable): T.Callable used to calculate
		similarities betweek queries and keys.
		Default is scaled_dot_product_similarity.
		pre_softmax (T.Callable): T.Callable returning a module
		executed immediately before softmax for transforming the
		attention values.
		Default is Identity.
		post_softmax (T.Callable): T.Callable returning a module
		executed immediately after softmax for transforming the
		attention values.
		weigh_values (T.Callable): T.Callable used to weigh
		the value vectors given the attention values.
		Default is jnp.matmul.
		proj_out (T.Callable): T.Callable returning a module that
		projects the weighed value vectors to the output dimension.
		Default is ProjOut.
	"""
	to_qkv: T.Optional[T.Union[T.Callable, int]] = None
	similarity_fn: T.Callable = scaled_dot_product_similarity
	pre_softmax: T.Callable = Identity
	post_softmax: T.Callable = Identity
	weigh_values: T.Callable = jnp.matmul
	proj_out: T.Callable = ProjOut

	@nn.compact
	def __call__(self, input=None, q=None, k=None, v=None):
		if input is not None:
			to_qkv = QKV(n_heads=self.to_qkv) if isinstance(self.to_qkv, int) else self.to_qkv()
			q, k, v = to_qkv(input)
		
		attention = self.similarity_fn(q, k)

		attention = self.pre_softmax()(attention)
		attention = nn.softmax(attention)
		attention = self.post_softmax()(attention)

		output = self.weigh_values(attention, v)
		output = self.proj_out()(output)
		
		return output
