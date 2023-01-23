"""
ConvNeXt.
"""


import typing as T

from flax import linen as nn

from .. import layers
from .factory import NORM_STATS, register_configs


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
	"""
	out_dim: T.Optional[int] = None
	layer_scale_init_value: T.Optional[float] = 1e-6
	grn: bool = False

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
			layer_scale_init_value=self.layer_scale_init_value,
			residual=False,
			)(output)
		return input+output


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
	"""
	depth: int
	out_dim: int
	stride: int = 1
	layer_scale_init_value: T.Optional[float] = 1e-6
	grn: bool = False

	@nn.compact
	def __call__(self, input):
		if (self.stride != 1) or (input.shape[-1] != self.out_dim):
			input = layers.PatchEmbed(
				token_dim=self.out_dim,
				patch_size=self.stride,
				layer_norm_eps=1e-6,
				norm_first=True,
				flatten=False,
				)(input)

		for _ in range(self.depth):
			input = ConvNeXtBlock(
				layer_scale_init_value=self.layer_scale_init_value,
				grn=self.grn,
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
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.out_dims[0],
			patch_size=4,
			layer_norm_eps=1e-6,
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


@register_configs
def get_convnext_configs() -> T.Tuple[T.Type[ConvNeXt], T.Dict]:
	"""
	Gets configurations for all available
	ConvNeXt models.

	Returns (T.Tuple[T.Type[ConvNeXt], T.Dict]): The ConvNeXt class and
	configurations of all available models.
	"""
	configs = {
		'convnext_xxxnano': {
			'depths': (2, 2, 6, 2),
			'out_dims': (40, 80, 160, 320),
			},
		'convnext_xxnano': {
			'depths': (2, 2, 6, 2),
			'out_dims': (48, 96, 192, 384),
			},
		'convnext_xnano': {
			'depths': (2, 2, 6, 2),
			'out_dims': (64, 128, 256, 512),
			},
		'convnext_nano': {
			'depths': (2, 2, 8, 2),
			'out_dims': (80, 160, 320, 640),
			},
		'convnext_tiny': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_small': {
			'depths': (3, 3, 27, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_base': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			},
		'convnext_large': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			},
		'convnext_nano_in12k': {
			'depths': (2, 2, 8, 2),
			'out_dims': (80, 160, 320, 640),
			},
		'convnext_tiny_in12k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_small_in12k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_nano_in12ft1k': {
			'depths': (2, 2, 8, 2),
			'out_dims': (80, 160, 320, 640),
			},
		'convnext_tiny_in12ft1k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_small_in12ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_tiny_in22k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_small_in22k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_base_in22k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			},
		'convnext_large_in22k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			},
		'convnext_xlarge_in22k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (256, 512, 1024, 2048),
			},
		'convnext_tiny_in22ft1k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_small_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_base_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			},
		'convnext_large_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			},
		'convnext_xlarge_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (256, 512, 1024, 2048),
			},
		'convnext_tiny_384_in22ft1k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_small_384_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (96, 192, 384, 768),
			},
		'convnext_base_384_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			},
		'convnext_large_384_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			},
		'convnext_xlarge_384_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (256, 512, 1024, 2048),
			},
		'convnextv2_atto_fcmae': {
			'depths': (2, 2, 6, 2),
			'out_dims': (40, 80, 160, 320),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_femto_fcmae': {
			'depths': (2, 2, 6, 2),
			'out_dims': (48, 96, 192, 384),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_pico_fcmae': {
			'depths': (2, 2, 6, 2),
			'out_dims': (64, 128, 256, 512),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_nano_fcmae': {
			'depths': (2, 2, 8, 2),
			'out_dims': (80, 160, 320, 640),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_tiny_fcmae': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_base_fcmae': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_large_fcmae': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_huge_fcmae': {
			'depths': (3, 3, 27, 3),
			'out_dims': (352, 704, 1408, 2816),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_atto_fcmae_ftin1k': {
			'depths': (2, 2, 6, 2),
			'out_dims': (40, 80, 160, 320),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_femto_fcmae_ftin1k': {
			'depths': (2, 2, 6, 2),
			'out_dims': (48, 96, 192, 384),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_pico_fcmae_ftin1k': {
			'depths': (2, 2, 6, 2),
			'out_dims': (64, 128, 256, 512),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_nano_fcmae_ftin1k': {
			'depths': (2, 2, 8, 2),
			'out_dims': (80, 160, 320, 640),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_tiny_fcmae_ftin1k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_base_fcmae_ftin1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_large_fcmae_ftin1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_huge_fcmae_ftin1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (352, 704, 1408, 2816),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_nano_fcmae_in22ft1k': {
			'depths': (2, 2, 8, 2),
			'out_dims': (80, 160, 320, 640),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_tiny_fcmae_in22ft1k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_base_fcmae_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_large_fcmae_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_nano_384_fcmae_in22ft1k': {
			'depths': (2, 2, 8, 2),
			'out_dims': (80, 160, 320, 640),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_tiny_384_fcmae_in22ft1k': {
			'depths': (3, 3, 9, 3),
			'out_dims': (96, 192, 384, 768),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_base_384_fcmae_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_large_384_fcmae_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (192, 384, 768, 1536),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_huge_384_fcmae_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (352, 704, 1408, 2816),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnextv2_huge_512_fcmae_in22ft1k': {
			'depths': (3, 3, 27, 3),
			'out_dims': (352, 704, 1408, 2816),
			'layer_scale_init_value': None,
			'grn': True,
			},
		'convnext_base_clip_laion2b': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'norm_stats': NORM_STATS['clip'],
			},
		'convnext_base_clip_laion2b_augreg': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'norm_stats': NORM_STATS['clip'],
			},
		'convnext_base_clip_laiona': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'norm_stats': NORM_STATS['clip'],
			},
		'convnext_base_clip_320_laiona': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'norm_stats': NORM_STATS['clip'],
			},
		'convnext_base_clip_320_laiona_augreg': {
			'depths': (3, 3, 27, 3),
			'out_dims': (128, 256, 512, 1024),
			'norm_stats': NORM_STATS['clip'],
			},
		}
	return ConvNeXt, configs
