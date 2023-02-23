"""
Class attention image transformer (Cait).
"""


import typing as T
from functools import partial

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from ..factory import imagenet_params_config, register_models


class TalkingHeads(nn.Module):
	"""
	Module enabling communication between different heads
	through a linear layer, a.k.a. talking heads.
	"""
	@nn.compact
	def __call__(self, input):
		output = jnp.transpose(input, (0, 2, 3, 1))
		output = nn.Dense(
			features=output.shape[-1],
			)(output)
		output = jnp.transpose(output, (0, 3, 1, 2))
		return output


class ClassAttentionQKV(nn.Module):
	"""
	Query-key-value extractor for class attention.

	Args:
		n_heads (int): Number of heads.
	"""
	n_heads: int

	@nn.compact
	def __call__(self, input):
		n_tokens, token_dim = input.shape[-2:]
		head_dim = token_dim//self.n_heads
		
		q = nn.Dense(
			features=token_dim,
			)(input[:, 0])
		kv = nn.Dense(
			features=2*token_dim,
			)(input)

		q = jnp.reshape(q, (-1, 1, self.n_heads, head_dim))
		kv = jnp.reshape(kv, (-1, n_tokens, 2, self.n_heads, head_dim))

		q = jnp.transpose(q, (0, 2, 1, 3))
		kv = jnp.transpose(kv, (2, 0, 3, 1, 4))
		k, v = jnp.split(
			ary=kv,
			indices_or_sections=2,
			axis=0,
			)

		return q, jnp.squeeze(k, axis=0), jnp.squeeze(v, axis=0)


class ClassAttentionBlock(nn.Module):
	"""
	Transformer block with class attention.

	Args:
		n_heads (int): Number of heads.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1e-5.
	"""
	n_heads: int
	layer_scale_init_value: T.Optional[float] = 1e-5

	@nn.compact
	def __call__(self, input, class_token):
		concat = jnp.concatenate((class_token, input), axis=1)
		output = nn.LayerNorm()(concat)
		output = layers.MHSA(
			to_qkv=partial(ClassAttentionQKV, n_heads=self.n_heads),
			)(output)
		output = layers.LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		class_token = class_token+output

		class_token = layers.TransformerMLP(
			layer_scale_init_value=self.layer_scale_init_value,
			)(class_token)

		return class_token


class Cait(nn.Module):
	"""
	Class-attention image transformer.

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
		Default is 1e-5.
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
	layer_scale_init_value: T.Optional[float] = 1e-5
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.token_dim,
			patch_size=self.patch_size,
			)(input)
		output = layers.AbsPosEmbed()(output)
		self.sow(
			col='intermediates',
			name='block_0',
			value=output,
			)

		for block_ind in range(self.depth):
			output = layers.MetaFormerBlock(
				token_mixer=partial(
					layers.MHSA,
					to_qkv=self.n_heads,
					pre_softmax=TalkingHeads,
					post_softmax=TalkingHeads,
					),
				layer_scale_init_value=self.layer_scale_init_value,
				)(output)
			self.sow(
				col='intermediates',
				name=f'block_{block_ind+1}',
				value=output,
				)

		class_token = layers.ClassToken(concat=False)(output)
		for block_ind in range(self.depth_class):
			class_token = ClassAttentionBlock(
				n_heads=self.n_heads,
				layer_scale_init_value=self.layer_scale_init_value,
				)(output, class_token)
			self.sow(
				col='intermediates',
				name=f'block_{self.depth+block_ind+1}',
				value=class_token,
				)
		
		output = jnp.concatenate((class_token, output), axis=1)
		output = layers.ViTHead(
			n_classes=self.n_classes,
			layer_norm_eps=1e-6,
			)(output)
			
		return output


@register_models
def get_cait_configs() -> T.Tuple[T.Type[Cait], T.Dict]:
	"""
	Gets configurations for all available
	Cait models.

	Returns (T.Tuple[T.Type[Cait], T.Dict]): The Cait class and
	configurations of all available models.
	"""
	configs = {
		'cait_xxsmall24': dict(
			model_args=dict(
				depth=24,
				token_dim=192,
				n_heads=4,
				),
			params={
				'in1k_224': imagenet_params_config('cait_xxsmall24_224'),
				'in1k_384': imagenet_params_config('cait_xxsmall24_384'),
				},
			),
		'cait_xxsmall36': dict(
			model_args=dict(
				depth=36,
				token_dim=192,
				n_heads=4,
				),
			params={
				'in1k_224': imagenet_params_config('cait_xxsmall36_224'),
				'in1k_384': imagenet_params_config('cait_xxsmall36_384'),
				},
			),
		'cait_xsmall24': dict(
			model_args=dict(
				depth=24,
				token_dim=288,
				n_heads=6,
				),
			params={
				'in1k_384': imagenet_params_config('cait_xsmall24_384'),
				},
			),
		'cait_small24': dict(
			model_args=dict(
				depth=24,
				token_dim=384,
				n_heads=8,
				),
			params={
				'in1k_224': imagenet_params_config('cait_small24_224'),
				'in1k_384': imagenet_params_config('cait_small24_384'),
				},
			),
		'cait_small36': dict(
			model_args=dict(
				depth=36,
				token_dim=384,
				n_heads=8,
				),
			params={
				'in1k_384': imagenet_params_config('cait_small36_384'),
				},
			),
		'cait_medium36': dict(
			model_args=dict(
				depth=36,
				token_dim=768,
				n_heads=16,
				layer_scale_init_value=1e-6,
				),
			params={
				'in1k_384': imagenet_params_config('cait_medium36_384'),
				},
			),
		'cait_medium48': dict(
			model_args=dict(
				depth=48,
				token_dim=768,
				n_heads=16,
				layer_scale_init_value=1e-6,
				),
			params={
				'in1k_448': imagenet_params_config('cait_medium48_448'),
				},
			),
		}
	return Cait, configs
