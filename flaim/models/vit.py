"""
Vision transformer (ViT).
"""


import typing as T
from functools import partial

from flax import linen as nn

from .. import layers
from .factory import NORM_STATS, register_configs


class ViT(nn.Module):
	"""
	Vision transformer.

	Args:
		depth (int): Depth.
		token_dim (int): Token dimension.
		n_heads (int): Number of heads.
		patch_size (int): Patch size. This value is used
		along both spatial dimensions.
		Default is 16.
		mlp_hidden_dim_expansion_factor (float): Factor of expansion
		for the hidden neurons of the MLP.
		Default is 4.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, no LayerScale
		is applied.
		Default is None.
		layer_norm_eps (float): Epsilon value passed to layer
		normalization.
		Default is 1e-6.
		k_bias (bool): Whether the linear transformation
		for obtaining keys should have a bias term.
		Default is True.
		rel_pos_embed (bool): Whether to use relative position
		embedding instead of absolute position embeddings. 
		If True, class_token_pos_embed is ignored.
		Default is False.
		pre_norm (bool): Whether to have layer normalization
		immediately before the body of the network.
		Default is False.
		class_token_pos_embed (bool): Whether to apply position 
		embedding to the class token.
		Default is True.
		n_classes (int): Number of output classes. If 0, there is no head,
		head_pool and head_norm_first are ignored, and the raw final features are returned. If -1,
		all stages of the head, other than the final linear layer, are applied
		and the output returned.
		Default is 0.
		head_pool (bool): Whether to average pool the tokens for generating
		predictions. If False, the class token is used to generate predictions.
		Default is False.
		head_norm_first (bool): Whether to apply layer normalization before
		pooling/class token extraction instead of after for the head.
		Default is True.
	"""
	depth: int
	token_dim: int
	n_heads: int
	patch_size: int = 16
	mlp_hidden_dim_expansion_factor: float = 4
	layer_scale_init_value: T.Optional[float] = None
	layer_norm_eps: float = 1e-6
	k_bias: bool = True
	rel_pos_embed: bool = False
	pre_norm: bool = False
	class_token_pos_embed: bool = True
	n_classes: int = 0
	head_pool: bool = False
	head_norm_first: bool = True

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.token_dim,
			patch_size=self.patch_size,
			bias=not self.pre_norm,
			)(input)
			
		if not self.rel_pos_embed:
			if self.class_token_pos_embed:
				output = layers.ClassToken()(output)
				output = layers.AbsPosEmbed()(output)
			
			else:
				output = layers.AbsPosEmbed()(output)
				output = layers.ClassToken()(output)
		
		else:
			output = layers.ClassToken()(output)
		
		output = nn.LayerNorm(
			epsilon=self.layer_norm_eps,
			)(output) if self.pre_norm else output
		self.sow(
			col='intermediates',
			name='block_0',
			value=output,
			)
		
		window_size = (input.shape[-3]//self.patch_size, input.shape[-2]//self.patch_size) if self.rel_pos_embed else None
		for block_ind in range(self.depth):
			output = layers.MetaFormerBlock(
				token_mixer=lambda: layers.MHSA(
					to_qkv=partial(
						layers.QKV,
						n_heads=self.n_heads,
						k_bias=self.k_bias,
						),
					pre_softmax=partial(
						layers.RelPosEmbed,
						n_heads=self.n_heads,
						window_size=window_size,
						) if window_size else layers.Identity,
					),
				mlp_hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
				layer_norm_eps=self.layer_norm_eps,
				layer_scale_init_value=self.layer_scale_init_value,
				)(output)
			self.sow(
				col='intermediates',
				name=f'block_{block_ind+1}',
				value=output,
				)

		if self.head_pool and self.n_classes != 0:
			output = output[:, 1:]
	
		output = layers.ViTHead(
			n_classes=self.n_classes,
			pool=self.head_pool,
			layer_norm_eps=self.layer_norm_eps,
			norm_first=self.head_norm_first,
			)(output)
		
		return output


@register_configs
def get_vit_configs() -> T.Tuple[T.Type[ViT], T.Dict]:
	"""
	Gets configurations for all available
	ViT models.

	Returns (T.Tuple[T.Type[ViT], T.Dict]): The ViT class and
	configurations of all available models.
	"""
	configs = {
		'vit_small_patch32_224': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch32_224': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_tiny_patch16_224': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 3,
			'norm_stats': NORM_STATS['inception'],
			}, 
		'vit_small_patch16_224': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'norm_stats': NORM_STATS['inception'],
			}, 
		'vit_base_patch16_224': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_large_patch16_224': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch8_224': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 8,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_small_patch32_384': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch32_384': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_large_patch32_384': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_tiny_patch16_384': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 3,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_small_patch16_384': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch16_384': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_large_patch16_384': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_small_patch32_224_in22k': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch32_224_in22k': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_large_patch32_224_in22k': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_tiny_patch16_224_in22k': {
			'depth': 12,
			'token_dim': 192,
			'n_heads': 3,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_small_patch16_224_in22k': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch16_224_in22k': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_large_patch16_224_in22k': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_huge_patch14_224_in22k': {
			'depth': 32,
			'token_dim': 1280,
			'n_heads': 16,
			'patch_size': 14,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch8_224_in22k': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 8,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch32_224_sam': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 32,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_base_patch16_224_sam': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 16,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_small_patch16_224_dino': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			},
		'vit_base_patch16_224_dino': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			},
		'vit_small_patch8_224_dino': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'patch_size': 8,
			},
		'vit_base_patch8_224_dino': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 8,
			},
		'vit_base_clip_patch32_224_laion2b': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 32,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['clip'],
			},
		'vit_base_clip_patch16_224_laion2b': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['clip'],
			},
		'vit_large_clip_patch14_224_laion2b': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'patch_size': 14,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['inception'],
			},
		'vit_huge_clip_patch14_224_laion2b': {
			'depth': 32,
			'token_dim': 1280,
			'n_heads': 16,
			'patch_size': 14,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['clip'],
			},
		'vit_giant_clip_patch14_224_laion2b': {
			'depth': 40,
			'token_dim': 1408,
			'n_heads': 16,
			'patch_size': 14,
			'mlp_hidden_dim_expansion_factor': 48/11,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['clip'],
			},
		'vit_base_clip_patch32_224_openai': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'patch_size': 32,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['clip'],
			},
		'vit_base_clip_patch16_224_openai': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['clip'],
			},
		'vit_large_clip_patch14_224_openai': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'patch_size': 14,
			'layer_norm_eps': 1e-5,
			'pre_norm': True,
			'norm_stats': NORM_STATS['clip'],
			},
		'deit3_small_patch16_224': {
			'depth': 12,
			'token_dim': 384, 
			'n_heads': 6,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_medium_patch16_224': {
			'depth': 12,
			'token_dim': 512,
			'n_heads': 8,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_base_patch16_224': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_large_patch16_224': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_huge_patch14_224': {
			'depth': 32,
			'token_dim': 1280,
			'n_heads': 16,
			'patch_size': 14,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_small_patch16_384': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_base_patch16_384': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_large_patch16_384': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_small_patch16_224_in22ft1k': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_medium_patch16_224_in22ft1k': {
			'depth': 12,
			'token_dim': 512,
			'n_heads': 8,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_base_patch16_224_in22ft1k': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_large_patch16_224_in22ft1k': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16, 
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_huge_patch14_224_in22ft1k': {
			'depth': 32,
			'token_dim': 1280,
			'n_heads': 16,
			'patch_size': 14,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_small_patch16_384_in22ft1k': {
			'depth': 12,
			'token_dim': 384,
			'n_heads': 6,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_base_patch16_384_in22ft1k': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'deit3_large_patch16_384_in22ft1k': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-6,
			'class_token_pos_embed': False,
			},
		'beit_base_patch16_224': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-1,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			'norm_stats': NORM_STATS['inception'],
			},
		'beit_large_patch16_224': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			'norm_stats': NORM_STATS['inception'],
			},
		'beit_base_patch16_384': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-1,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			'norm_stats': NORM_STATS['inception'],
			},
		'beit_large_patch16_384': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			'norm_stats': NORM_STATS['inception'],
			},
		'beit_large_patch16_512': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			'norm_stats': NORM_STATS['inception'],
			},
		'beit_base_patch16_224_in22k': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-1,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			'norm_stats': NORM_STATS['inception'],
			},
		'beit_large_patch16_224_in22k': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			'norm_stats': NORM_STATS['inception'],
			},
		'beitv2_base_patch16_224': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			},
		'beitv2_large_patch16_224': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			},
		'beitv2_base_patch16_224_in22k': {
			'depth': 12,
			'token_dim': 768,
			'n_heads': 12,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			},
		'beitv2_large_patch16_224_in22k': {
			'depth': 24,
			'token_dim': 1024,
			'n_heads': 16,
			'layer_scale_init_value': 1e-5,
			'rel_pos_embed': True,
			'k_bias': False,
			'head_pool': True,
			'head_norm_first': False,
			},
		}
	return ViT, configs
