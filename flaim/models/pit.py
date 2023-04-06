"""
Pooling-based vision transformer (PiT).
"""


import typing as T
from functools import partial

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from ..factory import imagenet_params_config, register_models


class PiTDownsample(nn.Module):
	"""
	PiT downsampling module.

	Args:
		out_dim (T.Optional[int]): Number of output channels.
		If None, it is set to 2*number of input channels.
		Default is None.
		stride (int): Stride.
		Default is 2.
	"""
	out_dim: T.Optional[int] = None
	stride: int = 2

	@nn.compact
	def __call__(self, input):
		class_token, input = input
		out_dim = self.out_dim or 2*input.shape[-1]

		output = layers.Conv(
			out_dim=out_dim,
			kernel_size=self.stride+1,
			stride=self.stride,
			groups='dw',
			)(input)
		class_token = nn.Dense(
			features=out_dim,
			)(class_token)

		return (class_token, output)


class PiTStage(nn.Module):
	"""
	PiT stage.

	Args:
		depth (int): Depth.
		n_heads (int): Number of heads.
		downsample (bool): Whether to downsample.
		Default is False.
	"""
	depth: int
	n_heads: int
	downsample: bool = False

	@nn.compact
	def __call__(self, input):
		class_token, input = input

		output = jnp.reshape(input, (len(input), -1, input.shape[-1]))
		output = jnp.concatenate((class_token, output), axis=1)
		for block_ind in range(self.depth):
			output = layers.MetaFormerBlock(
				token_mixer=partial(layers.MHSA, to_qkv=self.n_heads),
				)(output)

		class_token, output = output[:, :1], output[:, 1:]
		output = jnp.reshape(output, input.shape)
		output = (class_token, output)

		if self.downsample:
			output = PiTDownsample()(output)

		return output


class PiTHead(nn.Module):
	"""
	PiT head.

	Args:
		n_classes (int): Number of output classes. If 0, the feature maps are returned.
		If -1, the class token, after applying normalization, is returned.
		Default is 0.
	"""
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		if self.n_classes == 0:
			return input[1]

		output = layers.ViTHead(
			n_classes=self.n_classes,
			layer_norm_eps=1e-6,
			)(input[0])

		return output


class PiT(nn.Module):
	"""
	Pooling-based vision transformer.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		token_dim (int): Token dimension.
		n_heads (int): Number of heads.
		patch_size (int): Patch size. This value is used along
		both spatial dimensions.
		Default is 16.
		patch_stride (int): Patch stride. This value is used along
		both spatial dimensions.
		Default is
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	token_dim: int
	n_heads: int
	patch_size: int = 16
	patch_stride: int = 8
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.Conv(
			out_dim=self.n_heads*self.token_dim,
			kernel_size=self.patch_size,
			stride=self.patch_stride,
			padding=0,
			)(input)
		output = layers.AbsPosEmbed(
			n_axes=-3,
			)(output)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		class_token = layers.ClassToken(concat=False)(output)
		output = (class_token, output)
		for stage_ind in range(len(self.depths)):
			output = PiTStage(
				depth=self.depths[stage_ind],
				n_heads=self.n_heads * (2 ** stage_ind),
				downsample=True if stage_ind < len(self.depths)-1 else False,
				)(output)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output[1],
				)

		output = PiTHead(
			n_classes=self.n_classes,
			)(output)

		return output


@register_models
def get_pit_configs() -> T.Tuple[T.Type[PiT], T.Dict]:
	"""
	Gets configurations for all available
	PiT models.

	Returns (T.Tuple[T.Type[PiT], T.Dict]): The PiT class and
	configurations of all available models.
	"""
	configs = {
		'pit_tiny': dict(
			model_args=dict(
				depths=(2, 6, 4),
				token_dim=32,
				n_heads=2,
				),
			params={
				'in1k_224': imagenet_params_config('pit_tiny_in1k_224'),
				},
			),
		'pit_xsmall': dict(
			model_args=dict(
				depths=(2, 6, 4),
				token_dim=48,
				n_heads=2,
				),
			params={
				'in1k_224': imagenet_params_config('pit_xsmall_in1k_224'),
				},
			),
		'pit_small': dict(
			model_args=dict(
				depths=(2, 6, 4),
				token_dim=48,
				n_heads=3,
				),
			params={
				'in1k_224': imagenet_params_config('pit_small_in1k_224'),
				},
			),
		'pit_base': dict(
			model_args=dict(
				depths=(3, 6, 4),
				token_dim=64,
				n_heads=4,
				patch_size=14,
				patch_stride=7,
				),
			params={
				'in1k_224': imagenet_params_config('pit_base_in1k_224'),
				},
			),
		}
	return PiT, configs
