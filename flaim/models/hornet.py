"""
High-order spatial interaction network (HorNet).
"""


import typing as T
from functools import partial

from flax import linen as nn

from .. import layers
from .factory import register_configs


class HorNetStage(nn.Module):
	"""
	HorNet stage.

	Args:
		depths (int): Depth.
		out_dim (int): Number of output channels.
		order (int): Order of g^n convolution.
		Default is 5.
		scale (float): Scale factor of g^n convolution.
		Default is 1/3.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1e-6.
		downsample (bool): Whether to downsample.
		Default is False.
	"""
	depth: int
	out_dim: int
	order: int = 5
	scale: float = 1/3
	layer_scale_init_value: T.Optional[float] = 1e-6
	downsample: bool = False

	@nn.compact
	def __call__(self, input):
		if self.downsample:
			input = layers.PatchEmbed(
				token_dim=self.out_dim,
				patch_size=2,
				layer_norm_eps=1e-6,
				norm_first=True,
				flatten=False,
				)(input)

		for _ in range(self.depth):
			input = layers.MetaFormerBlock(
				token_mixer=partial(
					layers.GnConv,
					out_dim=self.out_dim,
					order=self.order,
					scale=self.scale,
					),
				layer_scale_init_value=self.layer_scale_init_value,
				)(input)

		return input


class HorNet(nn.Module):
	"""
	High-order spatial interaction network.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		out_dim_first_stage (int): Number of output channels of the first stage.
		orders (T.Tuple[int, ...]): Order of g^n convolution of each stage.
		Default is (2, 3, 4, 5).
		scale (float): Scale factor of g^n convolution.
		Default is 1/3.
		layer_scale_init_value (T.Optional[float]): Value for initializing
		LayerScale. If None, no LayerScale is applied.
		Default is 1e-6.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	out_dim_first_stage: int
	orders: T.Tuple[int, ...] = (2, 3, 4, 5)
	scale: float = 1/3
	layer_scale_init_value: T.Optional[float] = 1e-6
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.out_dim_first_stage,
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
			output = HorNetStage(
				depth=self.depths[stage_ind],
				out_dim=self.out_dim_first_stage * (2**stage_ind),
				order=self.orders[stage_ind],
				scale=self.scale,
				layer_scale_init_value=self.layer_scale_init_value,
				downsample=False if stage_ind == 0 else True,
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
def get_hornet_configs() -> T.Tuple[T.Type[HorNet], T.Dict]:
	"""
	Gets configurations for all available
	HorNet models.

	Returns (T.Tuple[T.Type[HorNet], T.Dict]): HorNet class and
	configurations of all models.
	"""
	configs = {
		'hornet_tiny': {
			'depths': (2, 3, 18, 2),
			'out_dim_first_stage': 64,
			}, 
		'hornet_small': {
			'depths': (2, 3, 18, 2),
			'out_dim_first_stage': 96,
			},
		'hornet_base': {
			'depths': (2, 3, 18, 2),
			'out_dim_first_stage': 128,
			}, 
		'hornet_large_in22k': {
			'depths': (2, 3, 18, 2),
			'out_dim_first_stage': 192,
			},
		}
	return HorNet, configs
