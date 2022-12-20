"""
Visual attention network (VAN).
"""


import typing as T

from flax import linen as nn

from .. import layers
from .factory import register_configs


class VANAttention(nn.Module):
	"""
	VAN attention module.

	Args:
		dw_kernel_size (int): Kernel size of
		the depthwise convolution.
		Default is 5.
		dilated_dw_kernel_size (int): Kernel size of
		the dilated depthwise convolution.
		Default is 7.
	"""
	dw_kernel_size: int = 5
	dilated_dw_kernel_size: int = 7

	@nn.compact
	def __call__(self, input):
		output = layers.Conv(
			kernel_size=1,
			)(input)
		output = layers.gelu(output)
		output = layers.LKA(
			dw_kernel_size=self.dw_kernel_size,
			dilated_dw_kernel_size=self.dilated_dw_kernel_size,
			)(output)
		output = layers.Conv(
			kernel_size=1,
			)(output)
		return input+output


class VANBlock(nn.Module):
	"""
	VAN block.

	Args:
		mlp_hidden_dim_expansion_factor (float): Factor of expansion
		for the hidden neurons of the MLP.
		Default is 4.
		dw_kernel_size (int): Kernel size of the depthwise convolution for LKA.
		Default is 5.
		dilated_dw_kernel_size (int): Kernel size of the dilated convolution for LKA.
		Default is 7.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, LayerScale is not
		applied.
		Default is 1e-2.
	"""
	mlp_hidden_dim_expansion_factor: float = 4.
	dw_kernel_size: int = 5
	dilated_dw_kernel_size: int = 7
	layer_scale_init_value: T.Optional[float] = 1e-2

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = nn.BatchNorm(
			use_running_average=not training,
			)(input)
		output = VANAttention(
			dw_kernel_size=self.dw_kernel_size,
			dilated_dw_kernel_size=self.dilated_dw_kernel_size,
			)(output)
		output = layers.LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		output = input+output

		residual = output
		output = nn.BatchNorm(
			use_running_average=not training,
			)(output)
		output = layers.TransformerMLP(
			hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
			dw_kernel_size=3,
			layer_norm_eps=None,
			layer_scale_init_value=self.layer_scale_init_value,
			residual=False,
			)(output)
		output = residual+output

		return output


class VANStage(nn.Module):
	"""
	VAN stage.

	Args:
		depth (int): Depth.
		out_dim (int): Number of output channels.
		mlp_hidden_dim_expansion_factor (float): Factor of expansion
		for the hidden neurons of the MLP.
		Default is 4.
		dw_kernel_size (int): Kernel size of the depthwise convolution for LKA.
		Default is 5.
		dilated_dw_kernel_size (int): Kernel size of the dilated depthwise convolution
		for LKA.
		Default is 7.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, LayerScale is not
		applied.
		downsample (bool): Whether there should be a downsampling
		module at the beginning.
		Default is False.
	"""
	depth: int
	out_dim: int
	mlp_hidden_dim_expansion_factor: float = 4.
	dw_kernel_size: int = 5
	dilated_dw_kernel_size: int = 7
	layer_scale_init_value: T.Optional[float] = 1e-2
	downsample: bool = False

	@nn.compact
	def __call__(self, input, training: bool = True):
		if self.downsample:
			input = layers.ConvBNAct(
				out_dim=self.out_dim,
				stride=2,
				bias_force=True,
				)(input, training=training)

		for _ in range(self.depth):
			input = VANBlock(
				dw_kernel_size=self.dw_kernel_size,
				dilated_dw_kernel_size=self.dilated_dw_kernel_size,
				mlp_hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
				layer_scale_init_value=self.layer_scale_init_value,
				)(input, training=training)
		output = nn.LayerNorm()(input)

		return output


class VAN(nn.Module):
	"""
	Visual attention network.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		out_dims (T.Tuple[int, ...]): Number of output channels of each stage.
		mlp_hidden_dim_expansion_factor (T.Tuple[float, ...]: Factor of expansion
		for the hidden neurons of the MLP of each stage.
		Default is (8, 8, 4, 4).
		dw_kernel_size (int): Kernel size of the depthwise convolution for LKA.
		Default is 5.
		dilated_dw_kernel_size (int): Kernel size of the dilated depthwise convolution
		for LKA.
		Default is 7.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, LayerScale is not
		applied.
		Default is 1e-2.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	out_dims: T.Tuple[int, ...]
	mlp_hidden_dim_expansion_factors: T.Tuple[float, ...] = (8., 8., 4., 4.)
	dw_kernel_size: int = 5
	dilated_dw_kernel_size: int = 7
	layer_scale_init_value: T.Optional[float] = 1e-2
	n_classes: int = 0

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.out_dims[0],
			kernel_size=7,
			stride=4,
			bias_force=True,
			)(input, training=training)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)
		
		for stage_ind in range(len(self.depths)):
			output = VANStage(
				depth=self.depths[stage_ind],
				out_dim=self.out_dims[stage_ind],
				mlp_hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factors[stage_ind],
				layer_scale_init_value=self.layer_scale_init_value,
				downsample=False if stage_ind == 0 else True,
				)(output, training=training)
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
def get_van_configs() -> T.Tuple[T.Type[VAN], T.Dict]:
	"""
	Gets configurations for all available
	VAN models.

	Returns (T.Tuple[T.Type[VAN], T.Dict]): The VAN class and
	configurations of all available models.
	"""
	configs = {
		'van_b0': {
			'depths': (3, 3, 5, 2),
			'out_dims': (32, 64, 160, 256),
			},
		'van_b1': {
			'depths': (2, 2, 4, 2),
			'out_dims': (64, 128, 320, 512),
			},
		'van_b2': {
			'depths': (3, 3, 12, 3),
			'out_dims': (64, 128, 320, 512),
			},
		'van_b3': {
			'depths': (3, 5, 27, 3),
			'out_dims': (64, 128, 320, 512),
			},
		}
	return VAN, configs
