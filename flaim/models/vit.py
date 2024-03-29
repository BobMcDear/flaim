"""
Vision transformer (ViT).
"""


import typing as T
from functools import partial

from flax import linen as nn

from .. import layers
from ..factory import clip_params_config, imagenet_params_config, inception_params_config, register_models


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


@register_models
def get_vit_configs() -> T.Tuple[T.Type[ViT], T.Dict]:
	"""
	Gets configurations for all available
	ViT models.

	Returns (T.Tuple[T.Type[ViT], T.Dict]): The ViT class and
	configurations of all available models.
	"""
	configs = {
		'vit_tiny_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=192,
				n_heads=3,
				),
			params={
				'augreg_in22k_224': inception_params_config('vit_tiny_patch16_224_in22k'),
				'augreg_in22k_ft_in1k_224': inception_params_config('vit_tiny_patch16_224'),
				'augreg_in22k_ft_in1k_384': inception_params_config('vit_tiny_patch16_384'),
				},
			),
		'vit_small_patch32': dict(
			model_args=dict(
				depth=12,
				token_dim=384,
				n_heads=6,
				patch_size=32,
				),
			params={
				'augreg_in22k_224': inception_params_config('vit_small_patch32_224_in22k'),
				'augreg_in22k_ft_in1k_224': inception_params_config('vit_small_patch32_224'),
				'augreg_in22k_ft_in1k_384': inception_params_config('vit_small_patch32_384'),
				},
			),
		'vit_small_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=384,
				n_heads=6,
				),
			params={
				'dino_in1k_224': imagenet_params_config('vit_small_patch16_224_dino'),
				'augreg_in1k_224': inception_params_config('vit_small_patch16_augreg_in1k_224'),
				'augreg_in1k_384': inception_params_config('vit_small_patch16_augreg_in1k_384'),
				'augreg_in22k_224': inception_params_config('vit_small_patch16_224_in22k'),
				'augreg_in22k_ft_in1k_224': inception_params_config('vit_small_patch16_224'),
				'augreg_in22k_ft_in1k_384': inception_params_config('vit_small_patch16_384'),
				},
			),
		'vit_small_patch8': dict(
			model_args=dict(
				depth=12,
				token_dim=384,
				n_heads=6,
				patch_size=8,
				),
			params={
				'dino_in1k_224': imagenet_params_config('vit_small_patch8_224_dino'),
				},
			),
		'vit_base_patch32': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				patch_size=32,
				),
			params={
				'sam_in1k_224': inception_params_config('vit_base_patch32_224_sam'),
				'augreg_in1k_224': inception_params_config('vit_base_patch32_augreg_in1k_224'),
				'augreg_in1k_384': inception_params_config('vit_base_patch32_augreg_in1k_384'),
				'augreg_in22k_224': inception_params_config('vit_base_patch32_224_in22k'),
				'augreg_in22k_ft_in1k_224': inception_params_config('vit_base_patch32_224'),
				'augreg_in22k_ft_in1k_384': inception_params_config('vit_base_patch32_384'),
				},
			),
		'vit_base_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				),
			params={
				'mae_in1k_224': imagenet_params_config('vit_base_patch16_mae_in1k_224'),
				'sam_in1k_224': inception_params_config('vit_base_patch16_224_sam'),
				'dino_in1k_224': imagenet_params_config('vit_base_patch16_224_dino'),
				'augreg_in1k_224': inception_params_config('vit_base_patch16_augreg_in1k_224'),
				'augreg_in1k_384': inception_params_config('vit_base_patch16_augreg_in1k_384'),
				'augreg_in22k_224': inception_params_config('vit_base_patch16_224_in22k'),
				'augreg_in22k_ft_in1k_224': inception_params_config('vit_base_patch16_224'),
				'augreg_in22k_ft_in1k_384': inception_params_config('vit_base_patch16_384'),
				},
			),
		'vit_base_pool_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				head_pool=True,
				head_norm_first=False,
				),
			params={
				'mae_in1k_ft_in1k_224': imagenet_params_config('vit_base_pool_patch16_mae_in1k_ft_in1k_224'),
				},
			),
		'vit_base_patch8': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				patch_size=8,
				),
			params={
				'dino_in1k_224': imagenet_params_config('vit_base_patch8_224_dino'),
				'augreg_in22k_224': inception_params_config('vit_base_patch8_224_in22k'),
				'augreg_in22k_ft_in1k_224': inception_params_config('vit_base_patch8_224'),
				},
			),
		'vit_large_patch32': dict(
			model_args=dict(
				depth=24,
				token_dim=1024,
				n_heads=16,
				patch_size=32,
				),
			params={
				'orig_in22k_224': inception_params_config('vit_large_patch32_224_in22k'),
				'orig_in22k_ft_in1k_384': inception_params_config('vit_large_patch32_384'),
				},
			),
		'vit_large_patch16': dict(
			model_args=dict(
				depth=24,
				token_dim=1024,
				n_heads=16,
				),
			params={
				'mae_in1k_224': imagenet_params_config('vit_large_patch16_mae_in1k_224'),
				'augreg_in22k_224': inception_params_config('vit_large_patch16_224_in22k'),
				'augreg_in22k_ft_in1k_224': inception_params_config('vit_large_patch16_224'),
				'augreg_in22k_ft_in1k_384': inception_params_config('vit_large_patch16_384'),
				},
			),
		'vit_large_pool_patch16': dict(
			model_args=dict(
				depth=24,
				token_dim=1024,
				n_heads=16,
				head_pool=True,
				head_norm_first=False,
				),
			params={
				'mae_in1k_ft_in1k_224': imagenet_params_config('vit_large_pool_patch16_mae_in1k_ft_in1k_224'),
				},
			),
		'vit_huge_patch14': dict(
			model_args=dict(
				depth=32,
				token_dim=1280,
				n_heads=16,
				patch_size=14,
				),
			params={
				'mae_in1k_224': imagenet_params_config('vit_huge_patch14_mae_in1k_224'),
				'orig_in22k_224': inception_params_config('vit_huge_patch14_224_in22k'),
				},
			),
		'vit_huge_pool_patch14': dict(
			model_args=dict(
				depth=32,
				token_dim=1280,
				n_heads=16,
				patch_size=14,
				head_pool=True,
				head_norm_first=False,
				),
			params={
				'mae_in1k_ft_in1k_224': imagenet_params_config('vit_huge_pool_patch14_mae_in1k_ft_in1k_224'),
				},
			),
		'vit_base_clip_patch32': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				patch_size=32,
				layer_norm_eps=1e-5,
				pre_norm=True,
				),
			params={
				'clip_openai_224': clip_params_config('vit_base_clip_patch32_224_openai'),
				'clip_openai_ft_in1k_224': clip_params_config('vit_base_clip_patch32_clip_openai_ft_in1k_224'),
				'clip_laion2b_224': clip_params_config('vit_base_clip_patch32_224_laion2b'),
				'clip_laion2b_ft_in1k_224': clip_params_config('vit_base_clip_patch32_clip_laion2b_ft_in1k_224'),
				},
			),
		'vit_base_clip_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				layer_norm_eps=1e-5,
				pre_norm=True,
				),
			params={
				'clip_openai_224': clip_params_config('vit_base_clip_patch16_224_openai'),
				'clip_openai_ft_in1k_224': clip_params_config('vit_base_clip_patch16_clip_openai_ft_in1k_224'),
				'clip_openai_ft_in1k_384': clip_params_config('vit_base_clip_patch16_clip_openai_ft_in1k_384'),
				'clip_laion2b_224': clip_params_config('vit_base_clip_patch16_224_laion2b'),
				'clip_laion2b_ft_in1k_224': clip_params_config('vit_base_clip_patch16_clip_laion2b_ft_in1k_224'),
				'clip_laion2b_ft_in1k_384': clip_params_config('vit_base_clip_patch16_clip_laion2b_ft_in1k_384'),
				},
			),
		'vit_large_clip_patch14': dict(
			model_args=dict(
				depth=24,
				token_dim=1024,
				n_heads=16,
				patch_size=14,
				layer_norm_eps=1e-5,
				pre_norm=True,
				),
			params={
				'clip_openai_224': clip_params_config('vit_large_clip_patch14_224_openai'),
				'clip_openai_ft_in1k_224': clip_params_config('vit_large_clip_patch14_clip_openai_ft_in1k_224'),
				'clip_laion2b_224': inception_params_config('vit_large_clip_patch14_224_laion2b'),
				'clip_laion2b_ft_in1k_224': inception_params_config('vit_large_clip_patch14_clip_laion2b_ft_in1k_224'),
				'clip_laion2b_ft_in1k_336': inception_params_config('vit_large_clip_patch14_clip_laion2b_ft_in1k_336'),
				},
			),
		'vit_huge_clip_patch14': dict(
			model_args=dict(
				depth=32,
				token_dim=1280,
				n_heads=16,
				patch_size=14,
				layer_norm_eps=1e-5,
				pre_norm=True,
				),
			params={
				'clip_laion2b_224': clip_params_config('vit_huge_clip_patch14_224_laion2b'),
				'clip_laion2b_ft_in1k_224': clip_params_config('vit_huge_clip_patch14_clip_laion2b_ft_in1k_224'),
				},
			),
		'vit_giant_clip_patch14': dict(
			model_args=dict(
				depth=40,
				token_dim=1408,
				n_heads=16,
				patch_size=14,
				mlp_hidden_dim_expansion_factor=48/11,
				layer_norm_eps=1e-5,
				pre_norm=True,
				),
			params={
				'clip_laion2b_224': clip_params_config('vit_giant_clip_patch14_224_laion2b'),
				},
			),
		'deit3_small_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=384,
				n_heads=6,
				layer_scale_init_value=1e-6,
				class_token_pos_embed=False,
				),
			params={
				'in1k_224': imagenet_params_config('deit3_small_patch16_224'),
				'in1k_384': imagenet_params_config('deit3_small_patch16_384'),
				'in22k_ft_in1k_224': imagenet_params_config('deit3_small_patch16_224_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('deit3_small_patch16_384_in22ft1k'),
				},
			),
		'deit3_medium_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=512,
				n_heads=8,
				layer_scale_init_value=1e-6,
				class_token_pos_embed=False,
				),
			params={
				'in1k_224': imagenet_params_config('deit3_medium_patch16_224'),
				'in22k_ft_in1k_224': imagenet_params_config('deit3_medium_patch16_224_in22ft1k'),
				},
			),
		'deit3_base_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				layer_scale_init_value=1e-6,
				class_token_pos_embed=False,
				),
			params={
				'in1k_224': imagenet_params_config('deit3_base_patch16_224'),
				'in1k_384': imagenet_params_config('deit3_base_patch16_384'),
				'in22k_ft_in1k_224': imagenet_params_config('deit3_base_patch16_224_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('deit3_base_patch16_384_in22ft1k'),
				},
			),
		'deit3_large_patch16': dict(
			model_args=dict(
				depth=24,
				token_dim=1024,
				n_heads=16,
				layer_scale_init_value=1e-6,
				class_token_pos_embed=False,
				),
			params={
				'in1k_224': imagenet_params_config('deit3_large_patch16_224'),
				'in1k_384': imagenet_params_config('deit3_large_patch16_384'),
				'in22k_ft_in1k_224': imagenet_params_config('deit3_large_patch16_224_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('deit3_large_patch16_384_in22ft1k'),
				},
			),
		'deit3_huge_patch14': dict(
			model_args=dict(
				depth=32,
				token_dim=1280,
				n_heads=16,
				patch_size=14,
				layer_scale_init_value=1e-6,
				class_token_pos_embed=False,
				),
			params={
				'in1k_224': imagenet_params_config('deit3_huge_patch14_224'),
				'in22k_ft_in1k_224': imagenet_params_config('deit3_huge_patch14_224_in22ft1k'),
				},
			),
		'beit_base_patch16': dict(
			model_args=dict(
				depth=12,
				token_dim=768,
				n_heads=12,
				layer_scale_init_value=1e-5,
				rel_pos_embed=True,
				k_bias=False,
				head_pool=True,
				head_norm_first=False,
				),
			params={
				'beit_in22k_ft_in22k_224': inception_params_config('beit_base_patch16_224_in22k'),
				'beit_in22k_ft_in22k_ft_in1k_224': inception_params_config('beit_base_patch16_224'),
				'beit_in22k_ft_in22k_ft_in1k_384': inception_params_config('beit_base_patch16_384'),
				'beitv2_in1k_ft_in22k_224': imagenet_params_config('beitv2_base_patch16_224_in22k'),
				'beitv2_in1k_ft_in22k_ft_in1k_224': imagenet_params_config('beitv2_base_patch16_224'),
				},
			),
		'beit_large_patch16': dict(
			model_args=dict(
				depth=24,
				token_dim=1024,
				n_heads=16,
				layer_scale_init_value=1e-5,
				rel_pos_embed=True,
				k_bias=False,
				head_pool=True,
				head_norm_first=False,
				),
			params={
				'beit_in22k_ft_in22k_224': inception_params_config('beit_large_patch16_224_in22k'),
				'beit_in22k_ft_in22k_ft_in1k_224': inception_params_config('beit_large_patch16_224'),
				'beit_in22k_ft_in22k_ft_in1k_384': inception_params_config('beit_large_patch16_384'),
				'beit_in22k_ft_in22k_ft_in1k_512': inception_params_config('beit_large_patch16_512'),
				'beitv2_in1k_ft_in22k_224': imagenet_params_config('beitv2_large_patch16_224_in22k'),
				'beitv2_in1k_ft_in22k_ft_in1k_224': imagenet_params_config('beitv2_large_patch16_224'),
				},
			),
		}
	return ViT, configs
