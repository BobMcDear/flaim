"""
Cross-covariance image transformer (XCiT).
"""


import typing as T
from functools import partial
from math import sqrt

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from .cait import ClassAttentionQKV
from .factory import register_configs


class DeepPatchEmbed(nn.Module):
	"""
	Deep patch embedding with multiple convolutions and non-linearities.

	Args:
		token_dim (int): Token dimension.
		patch_size (int): Patch size. This value is used along both
		spatial dimensions.
		Default is 16.
		act (T.Callable): Activation function.
		Default is layers.gelu.
	"""
	token_dim: int
	patch_size: int = 16
	act: T.Callable = layers.gelu

	@nn.compact
	def __call__(self, input, training: bool = False):
		if self.patch_size in [8, 16]:
			if self.patch_size == 16:
				input = layers.ConvBNAct(
					out_dim=self.token_dim//8,
					stride=2,
					act=self.act,
					)(input, training=training)
		
			output = layers.ConvBNAct(
					out_dim=self.token_dim//4,
					stride=2,
					act=self.act,
					)(input, training=training)
			output = layers.ConvBNAct(
					out_dim=self.token_dim//2,
					stride=2,
					act=self.act,
					)(output, training=training)
			output = layers.ConvBNAct(
				out_dim=self.token_dim,
				stride=2,
				)(output, training=training)
		
		output = jnp.reshape(output, (len(input), -1, self.token_dim))
		return output


class XCiTSinePosEmbed(nn.Module):
	"""
	Sinusoidal position embedding for XCiT.

	Args:
		hidden_dim (int): Dimension at which the position embeddings
		are generated. They are ultimately reverted to the token dimension.
		Default is 32.
		temp (float): Temperature.
		Default is 1e4.
		eps (float): Epsilon value for the denominator.
		Default is 1e-6.
		add (bool): Whether to add position embedding to the input
		when returning the output. If False, only the position embedding
		is returned.
		Default is True.
	"""
	hidden_dim: int = 32
	temp: float = 1e4
	eps: float = 1e-6
	add: bool = True

	@nn.compact
	def __call__(self, input):
		bs, n_tokens, token_dim = input.shape
		img_size = int(sqrt(n_tokens))

		pos = jnp.arange(1, img_size+1, dtype=jnp.float32)
		pos_x = jnp.reshape(pos, (1, 1, img_size))
		pos_x = jnp.repeat(pos_x, repeats=img_size, axis=1)
		pos_y = jnp.reshape(pos, (1, img_size, 1))
		pos_y = jnp.repeat(pos_y, repeats=img_size, axis=2)

		scale = 2*jnp.pi
		pos_embed_x = scale * pos_x / (pos_x[:, :, -1:] + self.eps)
		pos_embed_y = scale * pos_y / (pos_y[:, -1:, :] + self.eps)

		t = jnp.arange(0, self.hidden_dim, dtype=jnp.float32)
		t = self.temp ** (2 * jnp.floor_divide(t, 2) / self.hidden_dim)

		pos_embed_x = pos_embed_x[:, :, :, None] / t
		pos_embed_y = pos_embed_y[:, :, :, None] / t

		pos_embed_x = jnp.stack([
			jnp.sin(pos_embed_x[:, :, :, 0::2]),
			jnp.cos(pos_embed_x[:, :, :, 1::2]),
			],
			axis=4)
		pos_embed_y = jnp.stack([
			jnp.sin(pos_embed_y[:, :, :, 0::2]),
			jnp.cos(pos_embed_y[:, :, :, 1::2]),
			],
			axis=4)
		
		pos_embed_x = jnp.reshape(pos_embed_x, (1, img_size, img_size, -1))
		pos_embed_y = jnp.reshape(pos_embed_y, (1, img_size, img_size, -1))
		pos_embed = jnp.concatenate([pos_embed_y, pos_embed_x], axis=3)

		pos_embed = layers.Conv(
			out_dim=token_dim,
			kernel_size=1,
			)(pos_embed)
		pos_embed = jnp.reshape(pos_embed, (1, n_tokens, token_dim))

		return pos_embed+input if self.add else pos_embed


class LPI(nn.Module):
	"""
	Local patch interaction module.

	Args:
		act (T.Callable): Activation function.
		Default is layers.gelu.
	"""
	act: T.Callable = layers.gelu

	@nn.compact
	def __call__(self, input, training: bool = True):
		bs, n_tokens, token_dim = input.shape
		img_size = int(sqrt(n_tokens))

		output = jnp.reshape(input, (bs, img_size, img_size, token_dim))
		output = layers.Conv(
			groups='dw',
			)(output)
		output = self.act(output)
		output = nn.BatchNorm(
			use_running_average=not training,
			)(output)
		output = layers.Conv(
			groups='dw',
			)(output)
		output = jnp.reshape(output, input.shape)
		
		return output


class XCiTBlock(nn.Module):
	"""
	XCiT block.

	Args:
		n_heads (int): Number of heads.
		act (T.Callable): Activation function.
		Default is layers.gelu.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, no LayerScale
		is applied.
		Default is 1.
	"""
	n_heads: int
	act: T.Callable = layers.gelu
	layer_scale_init_value: T.Optional[float] = 1.

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = nn.LayerNorm()(input)
		output = layers.XCA(
			n_heads=self.n_heads,
			)(output)
		output = layers.LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		output = input+output

		residual = output
		output = nn.LayerNorm()(output)
		output = LPI(
			act=self.act,
			)(output, training=training)
		output = layers.LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		output = residual+output

		output = layers.TransformerMLP(
			act=self.act,
			layer_scale_init_value=self.layer_scale_init_value,
			)(output)
		
		return output


class XCiTClassAttentionBlock(nn.Module):
	"""
	Transformer block with class attention for XCiT.

	Args:
		n_heads (int): Number of heads.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1.
		norm_all_tokens (bool): Whether to apply normalization to all
		tokens and not merely the class token.
		Default is True.
	"""
	n_heads: int
	layer_scale_init_value: T.Optional[float] = 1.
	norm_all_tokens: bool = True

	@nn.compact
	def __call__(self, input):
		output = nn.LayerNorm()(input)
		class_token = layers.MHSA(
			to_qkv=partial(ClassAttentionQKV, n_heads=self.n_heads),
			)(output)
		output = jnp.concatenate([class_token, output[:, 1:]], axis=1)
		output = layers.LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		output = input+output

		if self.norm_all_tokens:
			output = nn.LayerNorm()(output)
		
		else:
			class_token = nn.LayerNorm()(output[:, :1])
			output = jnp.concatenate([class_token, output[:, 1:]], axis=1)

		residual = output
		class_token = layers.TransformerMLP(
			layer_norm_eps=None,
			layer_scale_init_value=self.layer_scale_init_value,
			residual=False,
			)(output[:, :1])
		output = jnp.concatenate([class_token, output[:, 1:]], axis=1)
		output = residual+output

		return output


class XCiT(nn.Module):
	"""
	Cross-covariance image transformer.

	Args:
		depth (int): Depth of the no-class-token 
		part of the model.
		token_dim (int): Token dimension.
		n_heads (int): Number of heads.
		patch_size (int): Patch size. This value
		is used along both spatial dimensions.
		Default is 16.
		depth_class (int): Depth of the class-token-only
		part of the model.
		Default is 2.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1.
		norm_all_tokens (bool): Whether to apply normalization to all
		tokens and not merely the class token in class attention.
		Default is True.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depth: int
	token_dim: int
	n_heads: int
	patch_size: int = 16
	depth_class: int = 2
	layer_scale_init_value: T.Optional[float] = 1.
	norm_all_tokens: bool = True
	n_classes: int = 0

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = DeepPatchEmbed(
			token_dim=self.token_dim,
			patch_size=self.patch_size,
			)(input, training=training)
		output = XCiTSinePosEmbed()(output)
		self.sow(
			col='intermediates',
			name='block_0',
			value=output,
			)

		for block_ind in range(self.depth):
			output = XCiTBlock(
				n_heads=self.n_heads,
				layer_scale_init_value=self.layer_scale_init_value,
				)(output, training=training)
			self.sow(
				col='intermediates',
				name=f'block_{block_ind+1}',
				value=output,
				)

		output = layers.ClassToken()(output)
		for block_ind in range(self.depth_class):
			output = XCiTClassAttentionBlock(
				n_heads=self.n_heads,
				layer_scale_init_value=self.layer_scale_init_value,
				norm_all_tokens=self.norm_all_tokens,
				)(output)
			self.sow(
				col='intermediates',
				name=f'block_{self.depth+block_ind+1}',
				value=output[:, 0],
				)
		
		output = layers.ViTHead(
			n_classes=self.n_classes,
			layer_norm_eps=1e-6,
			)(output)
			
		return output


@register_configs
def get_xcit_configs() -> T.Tuple[T.Type[XCiT], T.Dict]:
	"""
	Gets configurations for all available
	XCiT models.

	Returns (T.Tuple[T.Type[XCiT], T.Dict]): The XCiT class and
	configurations of all available models.
	"""
	configs = {
		'xcit_nano12_patch16_224': {
			'depth': 12,
			'token_dim': 128,
			'n_heads': 4,
			'norm_all_tokens': False,
			},
		'xcit_nano12_patch16_224_dist': {
			'depth': 12,
			'token_dim': 128,
			'n_heads': 4,
			'norm_all_tokens': False,
			},
		'xcit_nano12_patch16_384_dist': {
			'depth': 12,
			'token_dim': 128,
			'n_heads': 4,
			'norm_all_tokens': False,
			},
		'xcit_nano12_patch8_224': {
			'depth': 12,
			'token_dim': 128,
			'n_heads': 4,
			'patch_size': 8,
			'norm_all_tokens': False,
			},
		'xcit_nano12_patch8_224_dist': {
			'depth': 12,
			'token_dim': 128,
			'n_heads': 4,
			'patch_size': 8,
			'norm_all_tokens': False,
			},
		'xcit_nano12_patch8_384_dist': {
			'depth': 12,
			'token_dim': 128,
			'n_heads': 4,
			'patch_size': 8,
			'norm_all_tokens': False,
			},
		'xcit_tiny12_patch16_224': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 4,
			},
		'xcit_tiny12_patch16_224_dist': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 4,
			},
		'xcit_tiny12_patch16_384_dist': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 4,
			},
		'xcit_tiny24_patch16_224': {
			'depth': 24,
			'token_dim': 192,
			'n_heads': 4,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_tiny24_patch16_224_dist': {
			'depth': 24,
			'token_dim': 192,
			'n_heads': 4,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_tiny24_patch16_384_dist': {
			'depth': 24,
			'token_dim': 192,
			'n_heads': 4,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_tiny12_patch8_224': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 4,
			'patch_size': 8,
			},
		'xcit_tiny12_patch8_224_dist': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 4,
			'patch_size': 8,
			},
		'xcit_tiny12_patch8_384_dist': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 4,
			'patch_size': 8,
			},
		'xcit_tiny24_patch8_224': {
			'depth': 24,
			'token_dim': 192,
			'n_heads': 4,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_tiny24_patch8_224_dist': {
			'depth': 24,
			'token_dim': 192,
			'n_heads': 4,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_tiny24_patch8_384_dist': {
			'depth': 24,
			'token_dim': 192,
			'n_heads': 4,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_small12_patch16_224': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 8,
			},
		'xcit_small12_patch16_224_dist': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 8,
			},
		'xcit_small12_patch16_384_dist': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 8,
			},
		'xcit_small24_patch16_224': {
			'depth': 24,
			'token_dim': 384,
			'n_heads': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_small24_patch16_224_dist': {
			'depth': 24,
			'token_dim': 384,
			'n_heads': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_small24_patch16_384_dist': {
			'depth': 24,
			'token_dim': 384,
			'n_heads': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_small12_patch8_224': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 8,
			'patch_size': 8,
			},
		'xcit_small12_patch8_224_dist': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 8,
			'patch_size': 8,
			},
		'xcit_small12_patch8_384_dist': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 8,
			'patch_size': 8,
			},
		'xcit_small24_patch8_224': {
			'depth': 24,
			'token_dim': 384,
			'n_heads': 8,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_small24_patch8_224_dist': {
			'depth': 24,
			'token_dim': 384,
			'n_heads': 8,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_small24_patch8_384_dist': {
			'depth': 24,
			'token_dim': 384,
			'n_heads': 8,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_medium24_patch16_224': {
			'depth': 24,
			'token_dim': 512,
			'n_heads': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_medium24_patch16_224_dist': {
			'depth': 24,
			'token_dim': 512,
			'n_heads': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_medium24_patch16_384_dist': {
			'depth': 24,
			'token_dim': 512,
			'n_heads': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_medium24_patch8_224': {
			'depth': 24,
			'token_dim': 512,
			'n_heads': 8,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_medium24_patch8_224_dist': {
			'depth': 24,
			'token_dim': 512,
			'n_heads': 8,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_medium24_patch8_384_dist': {
			'depth': 24,
			'token_dim': 512,
			'n_heads': 8,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_large24_patch16_224': {
			'depth': 24,
			'token_dim': 768,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_large24_patch16_224_dist': {
			'depth': 24,
			'token_dim': 768,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_large24_patch16_384_dist': {
			'depth': 24,
			'token_dim': 768,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_large24_patch8_224': {
			'depth': 24,
			'token_dim': 768,
			'n_heads': 16,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_large24_patch8_224_dist': {
			'depth': 24,
			'token_dim': 768,
			'n_heads': 16,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		'xcit_large24_patch8_384_dist': {
			'depth': 24,
			'token_dim': 768,
			'n_heads': 16,
			'patch_size': 8,
			'layer_scale_init_value': 1e-5,
			},
		}
	return XCiT, configs
