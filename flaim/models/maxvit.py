"""
Multi-axis vision transformer (MaxViT).
"""


import typing as T
from functools import partial

import jax
from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from .factory import NORM_STATS, register_configs


class MaxViTStem(nn.Module):
	"""
	MaxViT stem.

	Args:
		out_dim (int): Number of output channels.
		Default is 64.
		tf (bool): Whether to use padding of 'same' and an
		approximation of GELU for compatibility with TensorFlow.
		Default is True.
	"""
	out_dim: int = 64
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			stride=2,
			bias_force=True,
			act=nn.gelu if self.tf else layers.gelu,
			tf=self.tf,
			)(input, training=training)
		output = layers.Conv(
			padding='same' if self.tf else 1,
			)(output)
		return output


class PreNormMBConvDownsample(nn.Module):
	"""
	PreNormMBConv downsampling module.

	Args:
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
	"""
	out_dim: int
	stride: int = 1

	@nn.compact
	def __call__(self, input):
		if self.stride != 1:
			input = layers.avg_pool(
				input=input,
				kernel_size=self.stride,
				stride=self.stride,
				padding=0,
				)
		
		if input.shape[-1] != self.out_dim:
			input = layers.Conv(
				out_dim=self.out_dim,
				kernel_size=1,
				)(input)
			
		return input


class PreNormMBConv(nn.Module):
	"""
	MBConv with normalization at the beginning.

	Args:
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
		expansion_factor (int): Expansion factor for
		the inverted bottleneck.
		Default is 4.
		act (T.Callable): Activation function.
		Default is nn.gelu.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
	"""
	out_dim: int
	stride: int = 1
	expansion_factor: int = 4
	act: T.Callable = nn.gelu
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = nn.BatchNorm(
			use_running_average=not training,
			epsilon=1e-3 if self.tf else 1e-5,
			)(input)

		output = layers.ConvBNAct(
			out_dim=4*self.out_dim,
			kernel_size=1,
			act=self.act,
			tf=self.tf,
			)(output, training=training)
		output = layers.ConvBNAct(
			stride=self.stride,
			groups='dw',
			act=self.act,
			tf=self.tf,
			)(output, training=training)
		output = layers.SE(
			reduction_factor=self.expansion_factor*4,
			act=nn.silu,
			)(output)
		output = layers.Conv(
			out_dim=self.out_dim,
			kernel_size=1,
			)(output)
		
		return output + PreNormMBConvDownsample(
			out_dim=self.out_dim,
			stride=self.stride,
			)(input)


def grid_partition(
	input,
	grid_size: T.Union[T.Tuple[int, int], int],
	):
	"""
	Partitions the input into grids.

	Args:
		input: Input.
		grid_size (T.Union[T.Tuple[int, int], int]): Grid size.
		If an int, this value is used along both spatial dimensions.
	
	Returns: The input partitioned into grids.
	"""
	bs, h, w, in_dim = input.shape
	grid_h, grid_w = layers.tuplify(grid_size)
	n_grids_h, n_grids_w = h//grid_h, w//grid_w

	output = jnp.reshape(input, (bs, grid_h, n_grids_h, grid_w, n_grids_w, in_dim))
	output = jnp.transpose(output, (0, 2, 4, 1, 3, 5))
	output = jnp.reshape(output, (-1, grid_h, grid_w, in_dim))
	return output


def grid_merge(
	input,
	img_size: T.Union[T.Tuple[int, int], int],
	grid_size: T.Union[T.Tuple[int, int], int],
	):
	"""
	Merges grids.

	Args:
		input: Input.
		img_size (T.Union[T.Tuple[int, int], int]): Image size.
		If an int, this value is used along both spatial dimensions.
		grid_size (T.Union[T.Tuple[int, int], int]): Grid size.
		If an int, this value is used along both spatial dimensions.
	
	Returns: The merged version of the input's grids.
	"""
	in_dim = input.shape[-1]
	img_h, img_w = layers.tuplify(img_size)
	grid_h, grid_w = layers.tuplify(grid_size)

	output = jnp.reshape(input, (-1, img_h//grid_h, img_w//grid_w, grid_h, grid_w, in_dim))
	output = jnp.transpose(output, (0, 3, 1, 4, 2, 5))
	output = jnp.reshape(output, (-1, img_h, img_w, in_dim))
	return output


def get_maxvit_rel_pos_ind(
	window_size: int,
	) -> jnp.ndarray:
	"""
	Gets a matrix used to index MaxViT's relative position bias
	table.

	Args:
		window_size (int): Window size. This value is used along
		both spatial dimesnoins.
	
	Returns (jnp.ndarray): Matrix used to index MaxViT's relative position bias
	table.
	"""
	max_rel_pos = window_size-1
	n_rel_distance = 2*max_rel_pos + 1

	rel_pos_ind = jnp.zeros((window_size, window_size, n_rel_distance))
	for i in range(window_size):
		for j in range(window_size):
			k = j - i + max_rel_pos
			if max_rel_pos < abs(j-i):
				continue
			rel_pos_ind = rel_pos_ind.at[i, j, k].set(1)

	return rel_pos_ind


def index_maxvit_rel_pos_table(
	rel_pos_table,
	rel_pos_ind,
	):
	"""
	Indexes a relative position bias table given a matrix of indices 
	for MaxViT.

	Args:
		rel_pos_table: Relative position bias table to index.
		rel_pos_ind: Matrix used to index rel_pos_table.

	Returns: Desired elements of rel_pos_table. 
	"""
	rel_pos_table = jnp.einsum('nhw,ixh->nixw', rel_pos_table, rel_pos_ind)
	rel_pos_table = jnp.einsum('nixw,jyw->nijxy', rel_pos_table, rel_pos_ind)

	area = len(rel_pos_ind) ** 2
	rel_pos_table = jnp.reshape(rel_pos_table, (len(rel_pos_table), area, area))
	return rel_pos_table


class MaxViTRelPosEmbed(nn.Module):
	"""
	Relative position embedding used by MaxViT.

	Args:
		n_heads (int): Number of heads.
		window_size (int): Window size. This value is used along
		both spatial dimesnoins.
	"""
	n_heads: int
	window_size: int

	@nn.compact
	def __call__(self, input):
		size = 2*self.window_size - 1
		rel_pos_table = self.param(
			name='rel_pos_table',
			init_fn=lambda prng: jax.random.normal(prng, (self.n_heads, size, size)),
			)
		rel_pos_ind = self.variable(
			col='rel_pos_ind',
			name='rel_pos_ind',
			init_fn=lambda: get_maxvit_rel_pos_ind(self.window_size),
			).value

		rel_pos_bias = index_maxvit_rel_pos_table(rel_pos_table, rel_pos_ind)
		return input+rel_pos_bias


class PartitionMHSA(nn.Module):
	"""
	Multi-headed self-attention, with support for
	partitioning.

	Args:
		partition_fn (T.Callable): T.Callable used to 
		partition the input.
		merge_fn (T.Callable): T.Callable used to
		merge the data after its partitioning.
		partition_size (int): Size of each partition.
		head_dim (int): Dimension per head.
		Default is 32.
	"""
	partition_fn: T.Callable
	merge_fn: T.Callable
	partition_size: int
	head_dim: int = 32

	@nn.compact
	def __call__(self, input):
		bs, h, w, in_dim = input.shape
		n_heads = in_dim//self.head_dim

		output = self.partition_fn(input, self.partition_size)
		output = jnp.reshape(output, (-1, self.partition_size ** 2, input.shape[-1]))

		output = layers.MHSA(
			to_qkv=n_heads,
			pre_softmax=partial(
				MaxViTRelPosEmbed,
				n_heads=n_heads,
				window_size=self.partition_size,
				),
			)(output)

		output = jnp.reshape(output, (-1, self.partition_size, self.partition_size, in_dim))
		output = self.merge_fn(output, (h, w), self.partition_size)

		return output


class MaxViTBlock(nn.Module):
	"""
	MaxViT block.

	Args:
		out_dim (int): Number of output channels.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		grid_size (int): Grid size for relative position
		embedding and grid attention. This value is used
		along both spatial dimensions.
		stride (int): Stride.
		Default is 1.
		head_dim (int): Dimension per head.
		Default is 32.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3, layer normalization epsilon of 1e-5, padding of 'same',
		and an approximation of GELU for compatibility with TensorFlow.
		Default is True.
	"""
	out_dim: int
	window_size: int
	grid_size: int
	stride: int = 1
	head_dim: int = 32
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		act = nn.gelu if self.tf else layers.gelu

		output = PreNormMBConv(
			out_dim=self.out_dim,
			stride=self.stride,
			act=act,
			tf=self.tf,
			)(input, training=training)
		output = layers.MetaFormerBlock(
			token_mixer=partial(
				PartitionMHSA,
				partition_fn=layers.window_partition,
				merge_fn=layers.window_merge,
				partition_size=self.window_size,
				head_dim=self.head_dim,
				),
			act=act,
			layer_norm_eps=1e-5 if self.tf else 1e-6,
			)(output)
		output = layers.MetaFormerBlock(
			token_mixer=partial(
				PartitionMHSA,
				partition_fn=grid_partition,
				merge_fn=grid_merge,
				partition_size=self.window_size,
				head_dim=self.head_dim,
				),
			act=act,
			layer_norm_eps=1e-5 if self.tf else 1e-6,
			)(output)

		return output


class MaxViTStage(nn.Module):
	"""
	MaxViT stage.

	Args:
		depth (int): Depth.
		out_dim (int): Number of output channels.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		grid_size (int): Grid size for relative position
		embedding and grid attention. This value is used
		along both spatial dimensions.
		stride (int): Stride.
		Default is 1.
		head_dim (int): Dimension per head.
		Default is 32.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3, layer normalization epsilon of 1e-5, padding of 'same',
		and an approximation of GELU for compatibility with TensorFlow.
		Default is True.
	"""
	depth: int
	out_dim: int
	window_size: int
	grid_size: int
	stride: int = 1
	head_dim: int = 32
	tf: bool = True
	
	@nn.compact
	def __call__(self, input, training: bool = False):
		for block_ind in range(self.depth):
			input = MaxViTBlock(
				out_dim=self.out_dim,
				stride=self.stride if block_ind == 0 else 1,
				window_size=self.window_size,
				grid_size=self.grid_size,
				head_dim=self.head_dim,
				tf=self.tf,
				)(input, training=training)
		return input


class MaxViTHead(nn.Module):
	"""
	MaxViT head.

	Args:
		n_classes (int): Number of output classes. If 0, the input is returned.
		If -1, all stages of the head, other than the final linear layer,
		are applied and the output returned. 
		Default is 0.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3, layer normalization epsilon of 1e-5, padding of 'same',
		and an approximation of GELU for compatibility with TensorFlow.
		Default is True.
	"""
	n_classes: int = 0
	tf: bool = True

	@nn.compact
	def __call__(self, input):
		if self.n_classes == 0:
			return input

		output = layers.Head(
			n_classes=input.shape[-1],
			layer_norm_eps=1e-5 if self.tf else 1e-6,
			)(input)
		output = nn.tanh(output)
		
		if self.n_classes != -1:
			output = nn.Dense(
				features=self.n_classes,
				)(output)
		
		return output


class MaxViT(nn.Module):
	"""
	Multi-axis vision transformer.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		out_dims (T.Tuple[int, ...]): Number of output channels of each stage.
		stem_out_dim (int): Number of output channels of the stem.
		Default is 64.
		partition_size_factor (int): Factor by which the spatial dimensions
		of the input are divided to obtain the partition size for window
		and grid attention.
		Default is 32.
		head_dim (int): Dimension per head.
		Default is 32.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3, layer normalization epsilon of 1e-5, padding of 'same',
		and an approximation of GELU for compatibility with TensorFlow.
		Default is True.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	out_dims: T.Tuple[int, ...]
	stem_out_dim: int = 64
	partition_size_factor: int = 32
	head_dim: int = 32
	tf: bool = True
	n_classes: int = 0

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = MaxViTStem(
			out_dim=self.stem_out_dim,
			)(input, training=training)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		partition_size = input.shape[-2]//self.partition_size_factor
		for stage_ind in range(len(self.depths)):
			output = MaxViTStage(
				depth=self.depths[stage_ind],
				out_dim=self.out_dims[stage_ind],
				stride=2,
				window_size=partition_size,
				grid_size=partition_size,
				head_dim=self.head_dim,
				tf=self.tf,
				)(output, training=training)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output,
				)
		
		output = MaxViTHead(
			n_classes=self.n_classes,
			)(output)
		
		return output


@register_configs
def get_maxvit_configs() -> T.Tuple[T.Type[MaxViT], T.Dict]:
	"""
	Gets configurations for all available
	MaxViT models.

	Returns (T.Tuple[T.Type[MaxViT], T.Dict]): The MaxViT class and
	configurations of all available models.
	"""
	configs = {
		'maxvit_tiny_224': {
			'depths': (2, 2, 5, 2),
			'out_dims': (64, 128, 256, 512),
			},
		'maxvit_small_224': {
			'depths': (2, 2, 5, 2),
			'out_dims': (96, 192, 384, 768),
			},
		'maxvit_base_224': {
			'depths': (2, 6, 14, 2),
			'out_dims': (96, 192, 384, 768),
			},
		'maxvit_large_224': {
			'depths': (2, 6, 14, 2),
			'out_dims': (128, 256, 512, 1024),
			'stem_out_dim': 128,
			},
		'maxvit_tiny_384': {
			'depths': (2, 2, 5, 2),
			'out_dims': (64, 128, 256, 512),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_small_384': {
			'depths': (2, 2, 5, 2),
			'out_dims': (96, 192, 384, 768),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_base_384': {
			'depths': (2, 6, 14, 2),
			'out_dims': (96, 192, 384, 768),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_large_384': {
			'depths': (2, 6, 14, 2),
			'out_dims': (128, 256, 512, 1024),
			'stem_out_dim': 128,
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_tiny_512': {
			'depths': (2, 2, 5, 2),
			'out_dims': (64, 128, 256, 512),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_small_512': {
			'depths': (2, 2, 5, 2),
			'out_dims': (96, 192, 384, 768),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_base_512': {
			'depths': (2, 6, 14, 2),
			'out_dims': (96, 192, 384, 768),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_large_512': {
			'depths': (2, 6, 14, 2),
			'out_dims': (128, 256, 512, 1024),
			'stem_out_dim': 128,
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_base_384_in22ft1k': {
			'depths': (2, 6, 14, 2),
			'out_dims': (96, 192, 384, 768),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_large_384_in22ft1k': {
			'depths': (2, 6, 14, 2),
			'out_dims': (128, 256, 512, 1024),
			'stem_out_dim': 128,
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_xlarge_384_in22ft1k': {
			'depths': (2, 6, 14, 2),
			'out_dims': (192, 384, 768, 1536),
			'stem_out_dim': 192,
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_base_512_in22ft1k': {
			'depths': (2, 6, 14, 2),
			'out_dims': (96, 192, 384, 768),
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_large_512_in22ft1k': {
			'depths': (2, 6, 14, 2),
			'out_dims': (128, 256, 512, 1024),
			'stem_out_dim': 128,
			'norm_stats': NORM_STATS['inception'],
			},
		'maxvit_xlarge_512_in22ft1k': {
			'depths': (2, 6, 14, 2),
			'out_dims': (192, 384, 768, 1536),
			'stem_out_dim': 192,
			'norm_stats': NORM_STATS['inception'],
			},
		}
	return MaxViT, configs
