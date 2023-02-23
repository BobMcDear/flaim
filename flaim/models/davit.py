"""
Dual attention vision transformer (DaViT).
"""


import typing as T

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from ..factory import imagenet_params_config, register_models


class ConvPosEnc(nn.Module):
	"""
	Position encoding with a convolution.
	"""
	@nn.compact
	def __call__(self, input):
		output = layers.Conv(
			groups='dw',
			)(input)
		return input+output


class DaViTSpatialBlock(nn.Module):
	"""
	DaViT spatial attention block.

	Args:
		n_heads (int): Number of heads.
		window_size (int): Window size for window attention.
		Default is 7.
	"""
	n_heads: int
	window_size: int = 7

	@nn.compact
	def __call__(self, input):
		residual = ConvPosEnc()(input)
		output = nn.LayerNorm(
			epsilon=1e-5,
			)(residual)
		output = layers.WindowMHSA(
			to_qkv=self.n_heads,
			window_size=self.window_size,
			)(output)
		output = residual+output

		output = ConvPosEnc()(output)
		output = layers.TransformerMLP(
			layer_norm_eps=1e-5,
			)(output)

		return output


class ChannelMHSA(nn.Module):
	"""
	Multi-headed self-attention applied over the channel axis.

	Args:
		n_heads (int): Number of heads.
	"""
	n_heads: int

	@nn.compact
	def __call__(self, input):
		q, k, v = layers.QKV(
			n_heads=self.n_heads,
			)(input)

		attention = jnp.swapaxes(k, axis1=-2, axis2=-1)  @ v / jnp.sqrt(k.shape[-1])
		attention = nn.softmax(attention)

		output = attention @ jnp.swapaxes(q, axis1=-2, axis2=-1)
		output = jnp.swapaxes(output, axis1=-2, axis2=-1)

		output = layers.ProjOut()(output)
		return output


class DaViTChannelBlock(nn.Module):
	"""
	DaViT channel attention block.

	Args:
		n_heads (int): Number of heads.
	"""
	n_heads: int

	@nn.compact
	def __call__(self, input):
		bs, h, w, in_dim = input.shape

		residual = ConvPosEnc()(input)
		residual = jnp.reshape(residual, (bs, -1, in_dim))
		output = nn.LayerNorm(
			epsilon=1e-5,
			)(residual)
		output = ChannelMHSA(
			n_heads=self.n_heads,
			)(output)
		output = residual+output

		output = jnp.reshape(output, (bs, h, w, in_dim))
		output = ConvPosEnc()(output)
		output = layers.TransformerMLP(
			layer_norm_eps=1e-5,
			)(output)

		return output


class DaViTStage(nn.Module):
	"""
	DaViT stage.

	Args:
		depth (int): Depth.
		out_dim (int): Number of output channels.
		n_heads (int): Number of heads.
		window_size (int): Window size for window attention.
		Default is 7.
		downsample (bool): Whether to downsample.
		Default is False.
	"""
	depth: int
	out_dim: int
	n_heads: int
	window_size: int = 7
	downsample: bool = False

	@nn.compact
	def __call__(self, input):
		if self.downsample:
			input = layers.PatchEmbed(
				token_dim=self.out_dim,
				patch_size=2,
				layer_norm_eps=1e-5,
				norm_first=True,
				flatten=False,
				)(input)

		for _ in range(self.depth):
			input = DaViTSpatialBlock(
				n_heads=self.n_heads,
				window_size=self.window_size,
				)(input)
			input = DaViTChannelBlock(
				n_heads=self.n_heads,
				)(input)

		return input


class DaViT(nn.Module):
	"""
	Dual attention vision transformer.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		out_dims (T.Tuple[int, ...]): Number of output channels of each stage.
		n_heads (T.Tuple[int, ...]): Number of heads of each stage.
		window_size (int): Window size for window attention.
		Default is 7.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	out_dims: T.Tuple[int, ...]
	n_heads: T.Tuple[int, ...]
	window_size: int = 7
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.out_dims[0],
			patch_size=7,
			patch_stride=4,
			layer_norm_eps=1e-5,
			flatten=False,
			)(input)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		for stage_ind in range(len(self.depths)):
			output = DaViTStage(
				depth=self.depths[stage_ind],
				out_dim=self.out_dims[stage_ind],
				n_heads=self.n_heads[stage_ind],
				window_size=self.window_size,
				downsample=False if stage_ind == 0 else True,
				)(output)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output,
				)

		output = layers.Head(
			n_classes=self.n_classes,
			layer_norm_eps=1e-5,
			)(output)

		return output


@register_models
def get_davit_configs() -> T.Tuple[T.Type[DaViT], T.Dict]:
	"""
	Gets configurations for all available
	DaViT models.

	Returns (T.Tuple[T.Type[DaViT], T.Dict]): The DaViT class and
	configurations of all available models.
	"""
	configs = {
		'davit_tiny': dict(
			model_args=dict(
				depths=(1, 1, 3, 1),
				out_dims=(96, 192, 384, 768),
				n_heads=(3, 6, 12, 24),
				),
			params={
				'in1k_224': imagenet_params_config('davit_tiny_224'),
				},
			),
		'davit_small': dict(
			model_args=dict(
				depths=(1, 1, 9, 1),
				out_dims=(96, 192, 384, 768),
				n_heads=(3, 6, 12, 24),
				),
			params={
				'in1k_224': imagenet_params_config('davit_small_224'),
				},
			),
		'davit_base': dict(
			model_args=dict(
				depths=(1, 1, 9, 1),
				out_dims=(128, 256, 512, 1024),
				n_heads=(4, 8, 16, 32),
				),
			params={
				'in1k_224': imagenet_params_config('davit_base_224'),
				},
			),
		}
	return DaViT, configs
