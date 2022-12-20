"""
Pyramid vision transformer V2 (PVT V2).
"""


import typing as T
from functools import partial
from math import sqrt

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from .factory import register_configs


class SRAQKV(nn.Module):
	"""
	Query-key-value extractor for spatial reduction attention. 

	Args:
		n_heads (int): Number of heads.
		sr_factor (int): Reduction factor for spatial
		downsampling of the key and value vectors.
		Default is 1.
	"""
	n_heads: int
	sr_factor: int = 1

	@nn.compact
	def __call__(self, input):
		input_kv = input
		token_dim = input.shape[-1]
		head_dim = token_dim//self.n_heads
		
		if 1 < self.sr_factor:
			img_size = int(sqrt(input.shape[-2]))
			input_kv = jnp.reshape(input, (-1, img_size, img_size, token_dim))
			input_kv = layers.PatchEmbed(
				token_dim=token_dim,
				patch_size=self.sr_factor,
				layer_norm_eps=1e-5,
				)(input_kv)
		
			q = nn.Dense(
				features=token_dim,
				)(input)
			kv = nn.Dense(
				features=2*token_dim,
				)(input_kv)
		
			q = jnp.reshape(q, (len(input), -1, self.n_heads, head_dim))
			kv = jnp.reshape(kv, (len(input), -1, 2, self.n_heads, head_dim))
		
			q = jnp.swapaxes(q, axis1=1, axis2=2)
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
				)(input)
		
		return q, k, v
		

class PVTV2Stage(nn.Module):
	"""
	PVTV2 stage.

	Args:
		depths (int): Depth.
		token_dim (int): Token dimension.
		n_heads (int): Number of heads.
		sr_factor (int): Reduction factor for spatial
		downsampling of the key and value vectors.
		Default is 1.
		mlp_hidden_dim_expansion_factor (float): Factor of expansion
		for the hidden neurons of the MLP.
		Default is 4.
		downsample (bool): Whether to downsample.
		Default is False.
	"""
	depth: int
	token_dim: int
	n_heads: int
	sr_factor: int = 1
	mlp_hidden_dim_expansion_factor: float = 4.
	downsample: bool = False

	@nn.compact
	def __call__(self, input):
		if self.downsample:
			input = layers.PatchEmbed(
				token_dim=self.token_dim,
				patch_size=3,
				patch_stride=2,
				layer_norm_eps=1e-5,
				)(input)

		for _ in range(self.depth):
			input = layers.MetaFormerBlock(
				token_mixer=lambda: layers.MHSA(
					to_qkv=partial(
						SRAQKV,
						n_heads=self.n_heads,
						sr_factor=self.sr_factor,
						),
					),
				mlp_hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
				dw_kernel_size=3,
				)(input)
		output = nn.LayerNorm()(input)

		img_size = int(sqrt(output.shape[-2]))
		output = jnp.reshape(output, (len(input), img_size, img_size, -1))
		return output


class PVTV2(nn.Module):
	"""
	Pyramid vision transformer V2.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		token_dims (T.Tuple[int, ...]): Token dimension of each stage.
		n_heads (T.Tuple[int, ...]): Number of heads of each
		stage.
		sr_factors (T.Tuple[int, ...]): Reduction factor for spatial
		reduction attention of each stage.
		Default is (8, 4, 2, 1).
		mlp_hidden_dim_expansion_factors (T.Tuple[float, ...]): Factor of expansion
		for the hidden neurons of the MLP of each stage.
		Default is (8, 8, 4, 4).
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	token_dims: T.Tuple[int, ...]
	n_heads: T.Tuple[int, ...]
	sr_factors: T.Tuple[int, ...] = (8, 4, 2, 1)
	mlp_hidden_dim_expansion_factors: T.Tuple[float, ...] = (8., 8., 4., 4.)
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.token_dims[0],
			patch_size=7,
			patch_stride=4,
			layer_norm_eps=1e-5,
			)(input)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)
		
		for stage_ind in range(len(self.depths)):
			output = PVTV2Stage(
				depth=self.depths[stage_ind],
				token_dim=self.token_dims[stage_ind],
				n_heads=self.n_heads[stage_ind],
				sr_factor=self.sr_factors[stage_ind],
				mlp_hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factors[stage_ind],
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
def get_pvtv2_configs() -> T.Tuple[T.Type[PVTV2], T.Dict]:
	"""
	Gets configurations for all available
	PVTV2 models.

	Returns (T.Tuple[T.Type[PVTV2], T.Dict]): PVTV2 class and
	configurations of all models.
	"""
	configs = {
		'pvtv2_b0': {
			'depths': (2, 2, 2, 2),
			'token_dims': (32, 64, 160, 256),
			'n_heads': (1, 2, 5, 8),
			},
		'pvtv2_b1': {
			'depths': (2, 2, 2, 2),
			'token_dims': (64, 128, 320, 512),
			'n_heads': (1, 2, 5, 8),
			},
		'pvtv2_b2': {
			'depths': (3, 4, 6, 3),
			'token_dims': (64, 128, 320, 512),
			'n_heads': (1, 2, 5, 8),
			},
		'pvtv2_b3': {
			'depths': (3, 4, 18, 3),
			'token_dims': (64, 128, 320, 512),
			'n_heads': (1, 2, 5, 8),
			},
		'pvtv2_b4': {
			'depths': (3, 8, 27, 3),
			'token_dims': (64, 128, 320, 512),
			'n_heads': (1, 2, 5, 8),
			},
		'pvtv2_b5': {
			'depths': (3, 6, 40, 3),
			'token_dims': (64, 128, 320, 512),
			'n_heads': (1, 2, 5, 8),
			'mlp_hidden_dim_expansion_factors': (4, 4, 4, 4),
			},
		}
	return PVTV2, configs
