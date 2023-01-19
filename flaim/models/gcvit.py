"""
Global context vision transformer (GC ViT).
"""


import typing as T
from functools import partial
from math import log2

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from .factory import register_configs


class GCViTFusedMBConv(nn.Module):
	"""
	Modified version of fused MBConv for GC ViT.

	Args:
		out_dim (T.Optional[int]): Number of output channels.
		If None, it is set to the number of input channels.
		Default is None.
		expansion_factor (int): Expansion factor for the
		width.
		Default is 1.
		act (T.Callable): Activation function.
		Default is layers.gelu.
	"""
	out_dim: T.Optional[int] = None
	expansion_factor: int = 1
	act: T.Callable = layers.gelu

	@nn.compact
	def __call__(self, input):
		in_dim = input.shape[-1]
		output = layers.ConvBNAct(
			out_dim=self.expansion_factor*in_dim,
			groups='dw',
			bias=False,
			bn=False,
			act=self.act,
			)(input)
		output = layers.SE(
			reduction_factor=4,
			act=layers.gelu,
			bias=False,
			)(output)
		output = layers.Conv(
			out_dim=self.out_dim or in_dim,
			kernel_size=1,
			bias=False,
			)(output)
		return input+output if input.shape == output.shape else output


class GCViTDownsample(nn.Module):
	"""
	GC ViT downsampling module.

	Args:
		out_dim (T.Optional[int]): Number of output channels.
		If None, it is set to the number of input channels.
		Default is None.
	"""
	out_dim: T.Optional[int] = None

	@nn.compact
	def __call__(self, input):
		output = nn.LayerNorm(
			epsilon=1e-5,
			)(input)
		output = GCViTFusedMBConv()(output)
		output = layers.Conv(
			out_dim=self.out_dim,
			stride=2,
			bias=False,
			)(output)
		output = nn.LayerNorm(
			epsilon=1e-5,
			)(output)
		return output

	
class GCViTStem(nn.Module):
	"""
	GC ViT stem.

	Args:
		out_dim (int): Number of output channels.
		Default is 96.
	"""
	out_dim: int = 96

	@nn.compact
	def __call__(self, input):
		output = layers.Conv(
			out_dim=self.out_dim,
			stride=2,
			)(input)
		output = GCViTDownsample()(output)
		return output


class GlobalQuery(nn.Module):
	"""
	Global query extractor.
	"""
	levels: int

	@nn.compact
	def __call__(self, input):
		reductions = self.levels
		for _ in range(max(1, self.levels)):
			input = GCViTFusedMBConv()(input)

			if reductions:
				input = layers.max_pool(input, stride=2)
				reductions -= 1

		return input


class WindowMHSA(nn.Module):
	"""
	Window multi-headed self-attention, with support for
	global queries.

	Args:
		n_heads (int): Number of heads.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		global_q (bool): Whether to use the provided global queries.
		Default is False.
	"""
	n_heads: int
	window_size: int
	global_q: bool = False

	@nn.compact
	def __call__(self, input, q_global=None):
		bs, h, w, in_dim = input.shape
		head_dim = in_dim//self.n_heads
		
		output = layers.window_partition(input, self.window_size)
		output = jnp.reshape(output, (-1, self.window_size**2, in_dim))

		if self.global_q:
			q = jnp.repeat(q_global, repeats=len(output)//len(q_global), axis=0)
			q = jnp.reshape(q, (len(q), -1, self.n_heads, head_dim))
			q = jnp.swapaxes(q, axis1=1, axis2=2)

			kv = nn.Dense(
				features=2*in_dim,
				)(output)
			kv = jnp.reshape(kv, (-1, self.window_size**2, 2, self.n_heads, head_dim))
			kv = jnp.transpose(kv, (2, 0, 3, 1, 4))
			k, v = jnp.split(
				ary=kv,
				indices_or_sections=2,
				axis=0,
				)
			k, v = jnp.squeeze(k, axis=0), jnp.squeeze(v, axis=0)
		
		else:
			q, k, v = layers.QKV(
				n_heads=self.n_heads,
				)(output)

		output = layers.MHSA(
			pre_softmax=partial(
				layers.RelPosEmbed,
				n_heads=self.n_heads,
				window_size=self.window_size, 
				class_token=False,
				),
			)(q=q, k=k, v=v)
		output = layers.window_merge(
			output,
			img_size=(h, w),
			window_size=self.window_size,
			)

		return output


class GCViTBlock(nn.Module):
	"""
	GC ViT block.

	Args:
		n_heads (int): Number of heads.
		global_q (bool): Whether to use the provided global queries.
		Default is False.
		mlp_hidden_dim_expansion_factor (float): Factor of expansion for the 
		hidden layer of the MLP.
		Default is 4.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, no LayerScale
		is applied.
		Default is None.
	"""
	n_heads: int
	global_q: bool = False
	mlp_hidden_dim_expansion_factor: int = 3
	window_size: int = 7
	layer_scale_init_value: T.Optional[float] = None

	@nn.compact
	def __call__(self, input, q_global=None):
		output = nn.LayerNorm(
			epsilon=1e-5,
			)(input)
		output = WindowMHSA(
			n_heads=self.n_heads,
			window_size=self.window_size,
			global_q=self.global_q,
			)(output, q_global=q_global)
		output = layers.LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		output = input+output

		output = layers.TransformerMLP(
			hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
			layer_norm_eps=1e-5,
			layer_scale_init_value=self.layer_scale_init_value,
			)(output)

		return output


class GCViTStage(nn.Module):
	"""
	GC ViT stage.

	Args:
		depth (int): Depth.
		n_heads (int): Number of heads.
		mlp_hidden_dim_expansion_factor (float): Factor of expansion for the 
		hidden layer of the MLP.
		Default is 4.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		Default is 7.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, no LayerScale
		is applied.
		Default is None.
		final_norm (bool): Whether to apply normalization on the output.
		Default is False.
		downsample (bool): Whether to downsample.
		Default is False.
	"""
	depth: int
	n_heads: int
	mlp_hidden_dim_expansion_factor: int = 3
	window_size: int = 7
	layer_scale_init_value: T.Optional[float] = None
	final_norm: bool = False
	downsample: bool = False

	@nn.compact
	def __call__(self, input):
		if self.downsample:
			input = GCViTDownsample(
				out_dim=2*input.shape[-1],
				)(input)
			
		q_global = GlobalQuery(
			levels=int(log2(input.shape[-2]//self.window_size)),
			)(input)
		for block_ind in range(self.depth):
			input = GCViTBlock(
				n_heads=self.n_heads,
				global_q=False if block_ind%2 == 0 else True,
				mlp_hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
				window_size=self.window_size,
				layer_scale_init_value=self.layer_scale_init_value,
				)(input, q_global)
			
		if self.final_norm:
			input = nn.LayerNorm(
				epsilon=1e-5,
				)(input)

		return input


class GCViT(nn.Module):
	"""
	Global context vision transformer.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		token_dim (int): Token dimension.
		n_heads (T.Tuple[int, ...]): Number of heads of each
		stage.
		mlp_hidden_dim_expansion_factor (float): Factor of expansion for the 
		hidden layer of the MLP.
		Default is 4.
		window_size_factors (T.Tuple[int, ...]): Factors by which
		the spatial dimensions of each stage are divided to obtain the window
		size.
		Default is (32, 32, 16, 32).
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, no LayerScale
		is applied.
		Default is None.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	token_dim: int
	n_heads: T.Tuple[int, ...]
	mlp_hidden_dim_expansion_factor: int = 3
	window_size_factors: T.Tuple[int, ...] = (32, 32, 16, 32)
	layer_scale_init_value: T.Optional[float] = None
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = GCViTStem(
			out_dim=self.token_dim,
			)(input)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		for stage_ind in range(len(self.depths)):
			output = GCViTStage(
				depth=self.depths[stage_ind],
				n_heads=self.n_heads[stage_ind],
				mlp_hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
				window_size=input.shape[-2]//self.window_size_factors[stage_ind],
				layer_scale_init_value=self.layer_scale_init_value,
				final_norm=(stage_ind == len(self.depths)-1),
				downsample=False if stage_ind == 0 else True,
				)(output)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output,
				)
		
		output = layers.Head(
			n_classes=self.n_classes,
			)(output)

		return output


@register_configs
def get_gcvit_configs() -> T.Tuple[T.Type[GCViT], T.Dict]:
	"""
	Gets configurations for all available
	Swin models.

	Returns (T.Tuple[T.Type[GCViT], T.Dict]): The GCViT class and
	configurations of all available models.
	"""
	configs = {
		'gcvit_xxtiny_224': {
			'depths': (2, 2, 6, 2),
			'token_dim': 64,
			'n_heads': (2, 4, 8, 16),
			},
		'gcvit_xtiny_224': {
			'depths': (3, 4, 6, 5),
			'token_dim': 64,
			'n_heads': (2, 4, 8, 16),
			},
		'gcvit_tiny_224': {
			'depths': (3, 4, 19, 5),
			'token_dim': 64,
			'n_heads': (2, 4, 8, 16),
			},
		'gcvit_small_224': {
			'depths': (3, 4, 19, 5),
			'token_dim': 96,
			'n_heads': (3, 6, 12, 24),
			'mlp_hidden_dim_expansion_factor': 2,
			'layer_scale_init_value': 1e-5,
			},
		'gcvit_base_224': {
			'depths': (3, 4, 19, 5),
			'token_dim': 128,
			'n_heads': (4, 8, 16, 32),
			'mlp_hidden_dim_expansion_factor': 2,
			'layer_scale_init_value': 1e-5,
			},
		}
	return GCViT, configs
