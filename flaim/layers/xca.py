"""
Cross-covariance attention (XCA) by El-Nouby et al.

References:
- El-Nouby et al. XCiT: Cross-Covariance Image Transformers.
"""


__all__ = [
	'XCA',
	]


from flax import linen as nn
from jax import numpy as jnp


class XCA(nn.Module):
	"""
	Cross-covariance attention.

	Args:
		n_heads (int): Number of heads.
		eps (float): Epsilon value for the denominator of the
		norm calculation.
		Default is 1e-12.
	"""
	n_heads: int
	eps: float = 1e-12

	@nn.compact
	def __call__(self, input):
		bs, n_tokens, token_dim = input.shape
		head_dim = token_dim//self.n_heads

		qkv = nn.Dense(
			features=3*token_dim,
			)(input)
		qkv = jnp.reshape(qkv, (-1, n_tokens, 3, self.n_heads, head_dim))
		qkv = jnp.transpose(qkv, (2, 0, 3, 4, 1))
		q, k, v = jnp.split(
			ary=qkv,
			indices_or_sections=3,
			axis=0,
			)
		q, k, v = jnp.squeeze(q, axis=0), jnp.squeeze(k, axis=0), jnp.squeeze(v, axis=0)
		
		norm_q, norm_k = jnp.linalg.norm(q, ord=2, axis=-1, keepdims=True), jnp.linalg.norm(k, ord=2, axis=-1, keepdims=True)
		norm_q, norm_k = jnp.clip(norm_q, a_min=self.eps), jnp.clip(norm_k, a_min=self.eps)
		q, k = q / norm_q, k / norm_k

		temp = self.param(
			name='temp',
			init_fn=lambda prng: jnp.ones((self.n_heads, 1, 1))
			)
		attention = q @ jnp.swapaxes(k, axis1=-2, axis2=-1)
		attention = nn.softmax(temp * attention)

		output = attention @ v
		output = jnp.transpose(output, (0, 3, 1, 2))
		output = jnp.reshape(output, input.shape)

		output = nn.Dense(
			features=token_dim,
			)(output)
		return output
