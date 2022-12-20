"""
Relative position embedding by Huang et al., code borrowed from Wightman.

References: 
- Huang et al. Music Transformer: Generating Music with Long-Term Structure.
- Wightman. PyTorch Image Models.
"""


__all__ = [
	'RelPosEmbed',
	]


import typing as T

from flax import linen as nn
from jax import numpy as jnp

from .tuplify import tuplify


def get_n_rel_distance(
	window_size: T.Union[T.Tuple[int, int], int],
	class_token: bool = True,
	) -> int:
	"""
	Gets the number of relative distances for
	relative position embedding.

	Args:
		window_size (T.Union[T.Tuple[int, int], int]): Window size.
		If an int, this value is used along both spatial dimesnoins.
		class_token (bool): Whether the input contains a class token.
		Default is True.
	"""
	window_size = tuplify(window_size)
	n_rel_distance = (2*window_size[0] - 1) * (2*window_size[1] - 1)
	return n_rel_distance+3 if class_token else n_rel_distance


def get_rel_pos_ind(
	window_size: T.Union[T.Tuple[int, int], int],
	class_token: bool = True,
	) -> jnp.ndarray:
	"""
	Gets a matrix used to index a relative position bias
	table.

	Args:
		window_size (T.Union[T.Tuple[int, int], int]): Window size.
		If an int, this value is used along both spatial dimesnoins.
		class_token (bool): Whether the input contains a class token.
		Default is True.
	
	Returns (jnp.ndarray): Matrix used to index a relative position bias
	table.
	"""
	window_h, window_w = tuplify(window_size)

	coords_h = jnp.arange(window_h)
	coords_w = jnp.arange(window_w)
	coords = jnp.meshgrid(coords_h, coords_w, indexing='ij')
	coords = jnp.stack(coords)
	coords = coords.reshape(len(coords), -1)

	rel_coords = coords[:, :, None] - coords[:, None, :]
	rel_coords = jnp.transpose(rel_coords, (1, 2, 0))
	
	rel_coords = rel_coords.at[:, :, 0].set(rel_coords[:, :, 0] + window_h - 1)
	rel_coords = rel_coords.at[:, :, 1].set(rel_coords[:, :, 1] + window_w - 1)
	rel_coords = rel_coords.at[:, :, 0].set(rel_coords[:, :, 0] * (2 * window_w - 1))

	if not class_token:
		return jnp.sum(rel_coords, -1)

	n_rel_distance = get_n_rel_distance(window_h, window_w)
	area = window_h*window_w

	rel_pos_ind = jnp.zeros(tuplify(area+1), dtype=int)
	rel_pos_ind = rel_pos_ind.at[1:, 1:].set(jnp.sum(rel_coords, -1))
	rel_pos_ind = rel_pos_ind.at[0, 0:].set(n_rel_distance-3)
	rel_pos_ind = rel_pos_ind.at[0:, 0].set(n_rel_distance-2)
	rel_pos_ind = rel_pos_ind.at[0, 0].set(n_rel_distance-1)

	return rel_pos_ind


class RelPosEmbed(nn.Module):
	"""
	Relative position embedding.

	Args:
		n_heads (int): Number of heads.
		window_size (T.Union[T.Tuple[int, int], int]): Window size.
		If an int, this value is used along both spatial dimesnoins.
		class_token (bool): Whether the input contains a class token.
		Default is True.
	"""
	n_heads: int
	window_size: T.Union[T.Tuple[int, int], int]
	class_token: bool = True

	@nn.compact
	def __call__(self, input):
		n_rel_distance = get_n_rel_distance(
			window_size=self.window_size,
			class_token=self.class_token,
			)
		rel_pos_table = nn.Embed(
			num_embeddings=n_rel_distance,
			features=self.n_heads,
			)
		rel_pos_ind = self.variable(
			col='rel_pos_ind',
			name='rel_pos_ind',
			init_fn=lambda: get_rel_pos_ind(
				window_size=self.window_size,
				class_token=self.class_token,
				),
			).value

		rel_pos_bias = rel_pos_table(rel_pos_ind)
		rel_pos_bias = jnp.transpose(rel_pos_bias, (2, 0, 1))
		
		return input+rel_pos_bias
