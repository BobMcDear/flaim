"""
Nested transformer (NesT).
"""


import typing as T
from functools import partial
from math import sqrt

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from .factory import register_configs


class ConvLNPool(nn.Module):
	"""
	Convolution followed by layer normalization and max pooling
	for downsampling.

	Args:
		token_dim (int): Token dimension.
		jax (bool): Whether padding should be fixed at 'same' for compatibility
		with JAX.
		Default is True.
	"""
	token_dim: int
	jax: bool = True

	@nn.compact
	def __call__(self, input):
		output = layers.ConvLNAct(
			out_dim=self.token_dim,
			kernel_size=3,
			padding='same' if self.jax else 1,
			)(input)
		output = layers.max_pool(
			input=output,
			stride=2,
			padding='same' if self.jax else 1,
			)
		return output


class BlockMHSA(nn.Module):
	"""
	Multi-headed self-attention with support for an extra
	block axis.

	Args:
		n_heads (int): Number of heads.
	"""
	n_heads: int

	@nn.compact
	def __call__(self, input):
		bs, n_blocks, n_tokens, token_dim = input.shape
		head_dim = token_dim//self.n_heads

		qkv = nn.Dense(
			features=3*token_dim,
			)(input)
		qkv = jnp.reshape(qkv, (bs, n_blocks, n_tokens, 3, self.n_heads, head_dim))
		qkv = jnp.transpose(qkv, (3, 0, 4, 1, 2, 5))
		q, k, v = jnp.split(
			ary=qkv,
			indices_or_sections=3,
			axis=0,
			)
		q, k, v = jnp.squeeze(q, axis=0), jnp.squeeze(k, axis=0), jnp.squeeze(v, axis=0)

		attention = q @ jnp.swapaxes(k, axis1=-2, axis2=-1) / jnp.sqrt(q.shape[-1])
		attention = nn.softmax(attention)

		output = attention @ v
		output = jnp.transpose(output, (0, 2, 3, 4, 1))
		output = jnp.reshape(output, (bs, n_blocks, n_tokens, token_dim))

		output = nn.Dense(
			features=output.shape[-1],
			)(output)

		return output


def block_partition(
	input,
	block_size: T.Union[T.Tuple[int, int], int],
	):
	"""
	Partitions the input into blocks.

	Args:
		input: Input.
		block_size (T.Union[T.Tuple[int, int], int]): Block size.
		If an int, this value is used along both spatial dimensions.
	
	Returns: The input partitioned into blocks.
	"""
	bs, h, w, in_dim = input.shape
	block_h, block_w = layers.tuplify(block_size)
	n_blocks_h, n_blocks_w = h//block_h, w//block_w

	output = jnp.reshape(input, (bs, n_blocks_h, block_h, n_blocks_w, block_w, in_dim))
	output = jnp.transpose(output, (0, 1, 3, 2, 4, 5))
	output = jnp.reshape(output, (bs, n_blocks_h*n_blocks_w, -1, in_dim))
	return output


def block_merge(
	input,
	block_size: T.Union[T.Tuple[int, int], int],
	):
	"""
	Merges blocks.

	Args:
		input: Input.
		block_size (T.Union[T.Tuple[int, int], int]): Block size.
		If an int, this value is used along both spatial dimensions.
	
	Returns: The input partitioned into blocks.
	"""
	bs, n_blocks, n_tokens, token_dim = input.shape
	block_h, block_w = layers.tuplify(block_size)
	n_blocks_h = n_blocks_w = int(sqrt(n_blocks))

	output = jnp.reshape(input, (bs, n_blocks_h, n_blocks_w, block_h, block_w, token_dim))
	output = jnp.transpose(output, (0, 1, 3, 2, 4, 5))
	output = jnp.reshape(output, (bs, block_h*n_blocks_h, block_w*n_blocks_w, token_dim))
	return output


class NesTStage(nn.Module):
	"""
	NesT stage, i.e., a single hierarchial level.

	Args:
		depth (int): Depth.
		token_dim (int): Token dimension.
		n_heads (int): Number of heads.
		block_size (T.Union[T.Tuple[int, int], int]): Block size.
		If an int, this value is used along both spatial dimensions.
		downsample (bool): Whether to downsample.
		Default is False.
		jax (bool): Whether padding should be fixed at 'same' for compatibility
		with JAX.
		Default is True.
	"""
	depth: int
	token_dim: int
	n_heads: int
	block_size: T.Union[T.Tuple[int, int], int] 
	downsample: bool = False
	jax: bool = True
	
	@nn.compact
	def __call__(self, input):
		if self.downsample:
			input = ConvLNPool(
				token_dim=self.token_dim,
				jax=self.jax,
				)(input)

		output = block_partition(input, self.block_size)
		output = layers.AbsPosEmbed(
			n_axes=-3,
			)(output)
		
		for _ in range(self.depth):
			output = layers.MetaFormerBlock(
				token_mixer=partial(BlockMHSA, n_heads=self.n_heads),
				)(output)
		output = block_merge(output, self.block_size) 

		return output


class NesT(nn.Module):
	"""
	Nested transformer.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		token_dims (T.Tuple[int, ...]): Token dimension of each stage.
		n_heads (T.Tuple[int, ...]): Number of heads of each
		stage.
		patch_size (int): Patch size. This value is used
		along both spatial dimensions.
		Default is 16.
		jax (bool): Whether padding should be fixed at 'same' for compatibility
		with JAX.
		Default is True.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	token_dims: T.Tuple[int, ...]
	n_heads: T.Tuple[int, ...]
	patch_size: int = 4
	jax: bool = True
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.token_dims[0],
			patch_size=self.patch_size,
			flatten=False,
			)(input)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		block_size = output.shape[-2] // int(sqrt(4 ** (len(self.depths)-1)))
		for stage_ind in range(len(self.depths)):
			output = NesTStage(
				depth=self.depths[stage_ind],
				token_dim=self.token_dims[stage_ind],
				n_heads=self.n_heads[stage_ind],
				block_size=block_size,
				downsample=False if stage_ind == 0 else True,
				jax=self.jax,
				)(output)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output,
				)
		
		output = layers.Head(
			n_classes=self.n_classes,
			layer_norm_eps=1e-6,
			norm_first=True,
			)(output)
		
		return output


@register_configs
def get_nest_configs() -> T.Tuple[T.Type[NesT], T.Dict]:
	"""
	Gets configurations for all available
	NesT models.

	Returns (T.Tuple[T.Type[NesT], T.Dict]): The NesT class and
	configurations of all available models.
	"""
	configs = {
		'nest_tiny_224': {
			'depths': (2, 2, 8),
			'token_dims': (96, 192, 384),
			'n_heads': (3, 6, 12),
			},
		'nest_small_224': {
			'depths': (2, 2, 20),
			'token_dims': (96, 192, 384),
			'n_heads': (3, 6, 12),
			},
		'nest_base_224': {
			'depths': (2, 2, 20),
			'token_dims': (128, 256, 512),
			'n_heads': (4, 8, 16),
			},
		}
	return NesT, configs
