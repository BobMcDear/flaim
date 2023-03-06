"""
ConvNeXt.
"""


import typing as T

from flax import linen as nn

from .. import layers
from ..factory import clip_params_config, imagenet_params_config, register_models


class ConvNeXtBlock(nn.Module):
	"""
	ConvNeXt block. 

	Args:
		out_dim (T.Optional[int]): Number of output channels.
		If None, it is set to the number of input channels.
		Default is None.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1e-6.
		grn (bool): Whether to use global response normalization inside the MLP.
		Default is False.
		layer_norm_eps (float): Epsilon value passed to layer
		normalization.
		Default is 1e-6.
	"""
	out_dim: T.Optional[int] = None
	layer_scale_init_value: T.Optional[float] = 1e-6
	grn: bool = False
	layer_norm_eps: float = 1e-6

	@nn.compact
	def __call__(self, input):
		output = layers.Conv(
			out_dim=self.out_dim,
			kernel_size=7,
			groups='dw',
			)(input)
		output = layers.TransformerMLP(
			act=nn.Sequential([
				layers.gelu,
				layers.GRN(),
				]) if self.grn else layers.gelu,
			layer_norm_eps=self.layer_norm_eps,
			layer_scale_init_value=self.layer_scale_init_value,
			residual=False,
			)(output)
		return input+output if input.shape == output.shape else output


class ConvNeXtStage(nn.Module):
	"""
	ConvNeXt stage.

	Args:
		depth (int): Depth.
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1e-6.
		grn (bool): Whether to use global response normalization inside the MLP.
		Default is False.
		layer_norm_eps (float): Epsilon value passed to layer
		normalization.
		Default is 1e-6.
	"""
	depth: int
	out_dim: int
	stride: int = 1
	layer_scale_init_value: T.Optional[float] = 1e-6
	grn: bool = False
	layer_norm_eps: float = 1e-6

	@nn.compact
	def __call__(self, input):
		if (self.stride != 1) or (input.shape[-1] != self.out_dim):
			input = layers.PatchEmbed(
				token_dim=self.out_dim,
				patch_size=self.stride,
				layer_norm_eps=self.layer_norm_eps,
				norm_first=True,
				flatten=False,
				)(input)

		for _ in range(self.depth):
			input = ConvNeXtBlock(
				layer_scale_init_value=self.layer_scale_init_value,
				grn=self.grn,
				layer_norm_eps=self.layer_norm_eps,
				)(input)

		return input


class ConvNeXt(nn.Module):
	"""
	ConvNeXt.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		out_dims (T.Tuple[int, ...]): Number of output channels of each stage.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1e-6.
		grn (bool): Whether to use global response normalization inside the MLP.
		Default is False.
		layer_norm_eps (float): Epsilon value passed to layer
		normalization.
		Default is 1e-6.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	out_dims: T.Tuple[int, ...]
	layer_scale_init_value: T.Optional[float] = 1e-6
	grn: bool = False
	layer_norm_eps: float = 1e-6
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.out_dims[0],
			patch_size=4,
			layer_norm_eps=self.layer_norm_eps,
			flatten=False,
			)(input)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		for stage_ind in range(len(self.depths)):
			output = ConvNeXtStage(
				depth=self.depths[stage_ind],
				out_dim=self.out_dims[stage_ind],
				stride=1 if stage_ind == 0 else 2,
				layer_scale_init_value=self.layer_scale_init_value,
				grn=self.grn,
				layer_norm_eps=self.layer_norm_eps,
				)(output)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output,
				)
		
		output = layers.Head(
			n_classes=self.n_classes,
			layer_norm_eps=1e-6,
			)(output)
			
		return output


@register_models
def get_convnext_configs() -> T.Tuple[T.Type[ConvNeXt], T.Dict]:
	"""
	Gets configurations for all available
	ConvNeXt models.

	Returns (T.Tuple[T.Type[ConvNeXt], T.Dict]): The ConvNeXt class and
	configurations of all available models.
	"""
	configs = {
		'convnext_atto': dict(
			model_args=dict(
				depths=(2, 2, 6, 2),
				out_dims=(40, 80, 160, 320),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_xxxnano'),
				},
			),
		'convnext_femto': dict(
			model_args=dict(
				depths=(2, 2, 6, 2),
				out_dims=(48, 96, 192, 384),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_xxnano'),
				},
			),
		'convnext_pico': dict(
			model_args=dict(
				depths=(2, 2, 6, 2),
				out_dims=(64, 128, 256, 512),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_xnano'),
				},
			),
		'convnext_nano': dict(
			model_args=dict(
				depths=(2, 2, 8, 2),
				out_dims=(80, 160, 320, 640),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_nano'),
				'in12k_224': imagenet_params_config('convnext_nano_in12k'),
				'in12k_ft_in1k_224': imagenet_params_config('convnext_nano_in12ft1k'),
				},
			),
		'convnext_tiny': dict(
			model_args=dict(
				depths=(3, 3, 9, 3),
				out_dims=(96, 192, 384, 768),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_tiny'),
				'in12k_224': imagenet_params_config('convnext_tiny_in12k'),
				'in12k_ft_in1k_224': imagenet_params_config('convnext_tiny_in12ft1k'),
				'in12k_ft_in1k_384': imagenet_params_config('convnext_tiny_384_in12ft1k'),
				'in22k_224': imagenet_params_config('convnext_tiny_in22k'),
				'in22k_ft_in1k_224': imagenet_params_config('convnext_tiny_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('convnext_tiny_384_in22ft1k'),
				},
			),
		'convnext_small': dict(
			model_args=dict(
				depths=(3, 3, 27, 3),
				out_dims=(96, 192, 384, 768),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_small'),
				'in12k_224': imagenet_params_config('convnext_small_in12k'),
				'in12k_ft_in1k_224': imagenet_params_config('convnext_small_in12ft1k'),
				'in12k_ft_in1k_384': imagenet_params_config('convnext_small_384_in12ft1k'),
				'in22k_224': imagenet_params_config('convnext_small_in22k'),
				'in22k_ft_in1k_224': imagenet_params_config('convnext_small_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('convnext_small_384_in22ft1k'),
				},
			),
		'convnext_base': dict(
			model_args=dict(
				depths=(3, 3, 27, 3),
				out_dims=(128, 256, 512, 1024),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_base'),
				'in22k_224': imagenet_params_config('convnext_base_in22k'),
				'in22k_ft_in1k_224': imagenet_params_config('convnext_base_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('convnext_base_384_in22ft1k'),
				'clip_laion2b_256': clip_params_config('convnext_base_clip_laion2b'),
				'clip_laion2b_augreg_256': clip_params_config('convnext_base_clip_laion2b_augreg'),
				'clip_laiona_256': clip_params_config('convnext_base_clip_laiona'),
				'clip_laiona_320': clip_params_config('convnext_base_clip_320_laiona'),
				'clip_laiona_augreg_320': clip_params_config('convnext_base_clip_320_laiona_augreg'),
				},
			),
		'convnext_large': dict(
			model_args=dict(
				depths=(3, 3, 27, 3),
				out_dims=(192, 384, 768, 1536),
				),
			params={
				'in1k_224': imagenet_params_config('convnext_large'),
				'in22k_224': imagenet_params_config('convnext_large_in22k'),
				'in22k_ft_in1k_224': imagenet_params_config('convnext_large_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('convnext_large_384_in22ft1k'),
				},
			),
		'convnext_xlarge': dict(
			model_args=dict(
				depths=(3, 3, 27, 3),
				out_dims=(256, 512, 1024, 2048),
				),
			params={
				'in22k_224': imagenet_params_config('convnext_xlarge_in22k'),
				'in22k_ft_in1k_224': imagenet_params_config('convnext_xlarge_in22ft1k'),
				'in22k_ft_in1k_384': imagenet_params_config('convnext_xlarge_384_in22ft1k'),
				},
			),
		'convnext_xxlarge': dict(
			model_args=dict(
				depths=(3, 4, 30, 3),
				out_dims=(384, 768, 1536, 3072),
				layer_norm_eps=1e-5,
				),
			params={
				'clip_laion2b_rewind_256': clip_params_config('convnext_xxlarge_clip_laion2b_rewind_256'),
				'clip_laion2b_soup_256': clip_params_config('convnext_xxlarge_clip_laion2b_soup_256'),
				},
			),
		'convnextv2_atto': dict(
			model_args=dict(
				depths=(2, 2, 6, 2),
				out_dims=(40, 80, 160, 320),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_atto_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_atto_fcmae_ftin1k'),
				},
			),
		'convnextv2_femto': dict(
			model_args=dict(
				depths=(2, 2, 6, 2),
				out_dims=(48, 96, 192, 384),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_femto_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_femto_fcmae_ftin1k'),
				},
			),
		'convnextv2_pico': dict(
			model_args=dict(
				depths=(2, 2, 6, 2),
				out_dims=(64, 128, 256, 512),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_pico_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_pico_fcmae_ftin1k'),
				},
			),
		'convnextv2_nano': dict(
			model_args=dict(
				depths=(2, 2, 8, 2),
				out_dims=(80, 160, 320, 640),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_nano_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_nano_fcmae_ftin1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_224': imagenet_params_config('convnextv2_nano_fcmae_in22ft1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_384': imagenet_params_config('convnextv2_nano_384_fcmae_in22ft1k'),
				},
			),
		'convnextv2_tiny': dict(
			model_args=dict(
				depths=(3, 3, 9, 3),
				out_dims=(96, 192, 384, 768),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_tiny_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_tiny_fcmae_ftin1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_224': imagenet_params_config('convnextv2_tiny_fcmae_in22ft1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_384': imagenet_params_config('convnextv2_tiny_384_fcmae_in22ft1k'),
				},
			),
		'convnextv2_base': dict(
			model_args=dict(
				depths=(3, 3, 27, 3),
				out_dims=(128, 256, 512, 1024),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_base_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_base_fcmae_ftin1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_224': imagenet_params_config('convnextv2_base_fcmae_in22ft1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_384': imagenet_params_config('convnextv2_base_384_fcmae_in22ft1k'),
				},
			),
		'convnextv2_large': dict(
			model_args=dict(
				depths=(3, 3, 27, 3),
				out_dims=(192, 384, 768, 1536),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_large_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_large_fcmae_ftin1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_224': imagenet_params_config('convnextv2_large_fcmae_in22ft1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_384': imagenet_params_config('convnextv2_large_384_fcmae_in22ft1k'),
				},
			),
		'convnextv2_huge': dict(
			model_args=dict(
				depths=(3, 3, 27, 3),
				out_dims=(352, 704, 1408, 2816),
				layer_scale_init_value=None,
				grn=True,
				),
			params={
				'fcmae_in1k_224': imagenet_params_config('convnextv2_huge_fcmae'),
				'fcmae_in1k_ft_in1k_224': imagenet_params_config('convnextv2_huge_fcmae_ftin1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_384': imagenet_params_config('convnextv2_huge_384_fcmae_in22ft1k'),
				'fcmae_in22k_ft_in22k_ft_in1k_512': imagenet_params_config('convnextv2_huge_512_fcmae_in22ft1k'),
				},
			),
		}
	return ConvNeXt, configs
