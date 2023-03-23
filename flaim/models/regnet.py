"""
RegNet.
"""


import typing as T
from functools import partial

import numpy as np
from flax import linen as nn

from .. import layers
from ..factory import imagenet_params_config, register_models
from .resnet import ResNetDownsample, ResNetStage


class RegNetBottleneckBlock(nn.Module):
	"""
	Residual bottleneck block for RegNet.

	Args:
		out_dim (int): Number of output channels.
		bottleneck_dim (int): Number of bottleneck channels.
		conv_bn_relu (T.Callable): Layer used as the 
		middle convolution-batch normalization-ReLU module.
		Default is layers.ConvBNAct.
		cardinality (int): Cardinality.
		Default is 1.
		stride (int): Stride.
		Default is 1.
		downsample (T.Callable): Downsampling module to use if input
		and output shapes are different.
		Default is ResNetDownsample.
		attention (T.Callable): T.Callable returning an attention module
		applied to the data.
		Default is Identity.
		attention_pre (bool): Whether to apply the attention module before or after
		the second 1 x 1 convolution.
		Default is True.
		attention_reduction_factor (int): Reduction factor for the attention module.
		The reduced dimension is calculated by dividing the number of input channels (not
		the number of intermediate channels) by this argument.
		Default is 4.
		avg_downsample (T.Optional[str]): If 'pre', average pooling is used
		before the 3 x 3 convolution for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If 'post', average pooling is used
		after the 3 x 3 convolution for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If None, spatial downsampling
		is performed by the 3 x 3 convolution.
		Default is None.
	"""
	out_dim: int
	bottleneck_dim: int
	conv_bn_relu: T.Callable = layers.ConvBNAct
	cardinality: int = 1
	stride: int = 1
	downsample: T.Callable = ResNetDownsample
	attention: T.Callable = layers.Identity
	attention_pre: bool = True
	attention_reduction_factor: int = 4
	avg_downsample: T.Optional[str] = None

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.bottleneck_dim,
			kernel_size=1,
			act=nn.relu,
			)(input, training=training)
		
		if self.avg_downsample == 'pre' and self.stride != 1:
			output = layers.avg_pool(
				input=output,
				stride=self.stride,
				)
		output = self.conv_bn_relu(
			stride=1 if self.avg_downsample else self.stride,
			groups=self.cardinality,
			act=nn.relu,
			)(output, training=training)
		if self.avg_downsample == 'post' and self.stride != 1:
			output = layers.avg_pool(
				input=output,
				stride=self.stride,
				)
		
		if self.attention_pre:
			output = self.attention(
				reduction_dim=input.shape[-1] // self.attention_reduction_factor,
				)(output)
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			kernel_size=1,
			)(output, training=training)
		if not self.attention_pre:
			output = self.attention(
				reduction_dim=input.shape[-1] // self.attention_reduction_factor,
				)(output)
		
		if input.shape != output.shape:
			input = self.downsample(
				out_dim=self.out_dim,
				stride=self.stride,
				)(input, training=training)
		
		output = input+output
		output = nn.relu(output)
		
		return output


class RegNet(nn.Module):
	"""
	RegNet.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		out_dims (T.Tuple[int, ...]): Number of output channels of each stage.
		dim_per_cardinal (int): Number of channels per cardinal group.
		conv_bn_relu (T.Callable): Layer used as the middle convolution-batch normalization-ReLU
		module of bottleneck blocks.
		Default is layers.ConvBNAct.
		bottleneck_factor (float): Bottleneck factor.
		Default is 1.
		stem (T.Callable): Stem.
		Default is partial(layers.ConvBNAct, stride=2, act=nn.relu).
		downsample (T.Callable): Downsampling module to use if input and output
		shapes are different.
		Default is ResNetDownsample.
		attention (T.Callable): T.Callable returning an attention module
		applied to the data.
		Default is Identity.
		attention_pre (bool): Whether to apply the attention module before or after
		the second 1 x 1 convolution.
		Default is True.
		avg_downsample (T.Optional[str]): If 'pre', average pooling is used
		before the first 3 x 3 convolution of each block for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If 'post', average pooling is used
		after the first 3 x 3 convolution of each block for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If None, spatial downsampling
		is performed by the first 3 x 3 convolution of each block.
		Default is None.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	out_dims: T.Tuple[int, ...]
	dim_per_cardinal: int
	conv_bn_relu: T.Callable = layers.ConvBNAct
	bottleneck_factor: float = 1.
	stem: T.Callable = partial(layers.ConvBNAct, stride=2, act=nn.relu)
	downsample: T.Callable = ResNetDownsample
	attention: T.Callable = layers.Identity
	attention_pre: bool = True
	avg_downsample: T.Optional[str] = None
	n_classes: int = 0

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = self.stem(
			out_dim=32,
			)(input, training=training)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		for stage_ind, depth in enumerate(self.depths):
			bottleneck_dim = int(self.out_dims[stage_ind] // self.bottleneck_factor)
			output = ResNetStage(
				depth=depth,
				out_dim=self.out_dims[stage_ind],
				bottleneck_dim=bottleneck_dim,
				block=RegNetBottleneckBlock,
				conv_bn_relu=self.conv_bn_relu,
				cardinality=max(bottleneck_dim // self.dim_per_cardinal, 1),
				stride=2,
				downsample=self.downsample,
				attention=self.attention,
				attention_pre=self.attention_pre,
				avg_downsample=self.avg_downsample,
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


def round_to_multiple(
	input,
	base: int = 8,
	):
	"""
	Rounds input to the nearest multiple of base.

	Args:
		input: Input.
		base (int): Base.
	
	Returns: Input rounded to the nearest multiple of base.
	"""
	return base * np.round(input/base).astype(int)


def get_regnet_args(
	depth: int,
	out_dim_slope: float,
	out_dim_base: int,
	log_step: float,
	dim_per_cardinal: int,
	) -> T.Dict:
	"""
	Gets the appropriate depth and number of output 
	channels per stage for RegNet.

	Args:
		depth (int): Depth.
		out_dim_slope (float): Progression slope for the number
		of channels.
		out_dim_base (int): The initial number of channels.
		log_step (float): Stepping size for the number of channels
		in log space.
		dim_per_cardinal (int): Number of channels per cardinal group.
	
	Returns (T.Dict): Dictionary containing the appropriate depth and
	number of output channels per stage for RegNet.
	"""
	out_dims_per_block_cont = out_dim_slope * np.arange(depth) + out_dim_base
	stage_ind = np.round(np.log(out_dims_per_block_cont / out_dim_base) / np.log(log_step))
	out_dims_per_block_quantized = round_to_multiple(out_dim_base * np.power(log_step, stage_ind))
	out_dims_per_stage, depths = np.unique(out_dims_per_block_quantized, return_counts=True)

	to_round = dim_per_cardinal<out_dims_per_stage
	out_dims_per_stage[to_round] = round_to_multiple(out_dims_per_stage[to_round], dim_per_cardinal)

	return {
		'depths': tuple(depths.tolist()),
		'out_dims': tuple(out_dims_per_stage.tolist()),
		'dim_per_cardinal': dim_per_cardinal,
		}


@register_models
def get_regnet_configs() -> T.Tuple[T.Type[RegNet], T.Dict]:
	"""
	Gets configurations for all available
	RegNet models.

	Returns (T.Tuple[T.Type[RegNet], T.Dict]): The RegNet class and
	configurations of all available models.
	"""
	configs = {
		'regnetx_200mf': dict(
			model_args=dict(
				**get_regnet_args(depth=13, out_dim_slope=36.44, out_dim_base=24, log_step=2.49, dim_per_cardinal=8),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_200mf'),
				},
			),
		'regnetx_400mf': dict(
			model_args=dict(
				**get_regnet_args(depth=22, out_dim_slope=24.48, out_dim_base=24, log_step=2.54, dim_per_cardinal=16),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_400mf'),
				},
			),
		'regnetx_600mf': dict(
			model_args=dict(
				**get_regnet_args(depth=16, out_dim_slope=36.97, out_dim_base=48, log_step=2.24, dim_per_cardinal=24),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_600mf'),
				},
			),
		'regnetx_800mf': dict(
			model_args=dict(
				**get_regnet_args(depth=16, out_dim_slope=35.73, out_dim_base=56, log_step=2.28, dim_per_cardinal=16),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_800mf'),
				},
			),
		'regnetx_1600mf': dict(
			model_args=dict(
				**get_regnet_args(depth=18, out_dim_slope=34.01, out_dim_base=80, log_step=2.25, dim_per_cardinal=24),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_1600mf'),
				},
			),
		'regnetx_3200mf': dict(
			model_args=dict(
				**get_regnet_args(depth=25, out_dim_slope=26.31, out_dim_base=88, log_step=2.25, dim_per_cardinal=48),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_3200mf'),
				},
			),
		'regnetx_4000mf': dict(
			model_args=dict(
				**get_regnet_args(depth=23, out_dim_slope=38.65, out_dim_base=96, log_step=2.43, dim_per_cardinal=40),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_4000mf'),
				},
			),
		'regnetx_6400mf': dict(
			model_args=dict(
				**get_regnet_args(depth=17, out_dim_slope=60.83, out_dim_base=184, log_step=2.07, dim_per_cardinal=56),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_6400mf'),
				},
			),
		'regnetx_8000mf': dict(
			model_args=dict(
				**get_regnet_args(depth=23, out_dim_slope=49.56, out_dim_base=80, log_step=2.88, dim_per_cardinal=120),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_8000mf'),
				},
			),
		'regnetx_12gf': dict(
			model_args=dict(
				**get_regnet_args(depth=19, out_dim_slope=73.36, out_dim_base=168, log_step=2.37, dim_per_cardinal=112),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_12gf'),
				},
			),
		'regnetx_16gf': dict(
			model_args=dict(
				**get_regnet_args(depth=22, out_dim_slope=55.59, out_dim_base=216, log_step=2.1, dim_per_cardinal=128),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_16gf'),
				},
			),
		'regnetx_32gf': dict(
			model_args=dict(
				**get_regnet_args(depth=23, out_dim_slope=69.86, out_dim_base=320, log_step=2., dim_per_cardinal=168),
				),
			params={
				'in1k_224': imagenet_params_config('regnetx_32gf'),
				},
			),
		'regnety_200mf': dict(
			model_args=dict(
				**get_regnet_args(depth=13, out_dim_slope=36.44, out_dim_base=24, log_step=2.49, dim_per_cardinal=8),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_200mf'),
				},
			),
		'regnety_400mf': dict(
			model_args=dict(
				**get_regnet_args(depth=16, out_dim_slope=27.89, out_dim_base=48, log_step=2.09, dim_per_cardinal=8),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_400mf'),
				},
			),
		'regnety_600mf': dict(
			model_args=dict(
				**get_regnet_args(depth=15, out_dim_slope=32.54, out_dim_base=48, log_step=2.32, dim_per_cardinal=16),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_600mf'),
				},
			),
		'regnety_800mf': dict(
			model_args=dict(
				**get_regnet_args(depth=14, out_dim_slope=38.84, out_dim_base=56, log_step=2.4, dim_per_cardinal=16),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_800mf'),
				},
			),
		'regnety_1600mf': dict(
			model_args=dict(
				**get_regnet_args(depth=27, out_dim_slope=20.71, out_dim_base=48, log_step=2.65, dim_per_cardinal=24),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_1600mf'),
				},
			),
		'regnety_3200mf': dict(
			model_args=dict(
				**get_regnet_args(depth=21, out_dim_slope=42.63, out_dim_base=80, log_step=2.66, dim_per_cardinal=24),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_3200mf'),
				},
			),
		'regnety_4000mf': dict(
			model_args=dict(
				**get_regnet_args(depth=22, out_dim_slope=31.41, out_dim_base=96, log_step=2.24, dim_per_cardinal=64),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_4000mf'),
				},
			),
		'regnety_6400mf': dict(
			model_args=dict(
				**get_regnet_args(depth=25, out_dim_slope=33.22, out_dim_base=112, log_step=2.27, dim_per_cardinal=72),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_6400mf'),
				},
			),
		'regnety_8000mf': dict(
			model_args=dict(
				**get_regnet_args(depth=17, out_dim_slope=76.82, out_dim_base=192, log_step=2.19, dim_per_cardinal=56),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_8000mf'),
				},
			),
		'regnety_12gf': dict(
			model_args=dict(
				**get_regnet_args(depth=19, out_dim_slope=73.36, out_dim_base=168, log_step=2.37, dim_per_cardinal=112),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_12gf'),
				},
			),
		'regnety_16gf': dict(
			model_args=dict(
				**get_regnet_args(depth=18, out_dim_slope=106.23, out_dim_base=200, log_step=2.48, dim_per_cardinal=112),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_16gf'),
				},
			),
		'regnety_32gf': dict(
			model_args=dict(
				**get_regnet_args(depth=20, out_dim_slope=115.89, out_dim_base=232, log_step=2.53, dim_per_cardinal=232),
				attention=layers.SE,
				),
			params={
				'in1k_224': imagenet_params_config('regnety_32gf'),
				'seer_224': imagenet_params_config('regnety_32gf_seer_224'),
				'seer_ft_in1k_384': imagenet_params_config('regnety_32gf_seer_ft_in1k_384'),
				},
			),
		'regnety_64gf': dict(
			model_args=dict(
				**get_regnet_args(depth=20, out_dim_slope=147.48, out_dim_base=352, log_step=2.4, dim_per_cardinal=328),
				attention=layers.SE,
				),
			params={
				'seer_224': imagenet_params_config('regnety_64gf_seer_224'),
				'seer_ft_in1k_384': imagenet_params_config('regnety_64gf_seer_ft_in1k_384'),
				},
			),
		'regnety_128gf': dict(
			model_args=dict(
				**get_regnet_args(depth=27, out_dim_slope=160.83, out_dim_base=456, log_step=2.52, dim_per_cardinal=264),
				attention=layers.SE,
				),
			params={
				'seer_224': imagenet_params_config('regnety_128gf_seer_224'),
				'seer_ft_in1k_384': imagenet_params_config('regnety_128gf_seer_ft_in1k_384'),
				},
			),
		'regnety_256gf': dict(
			model_args=dict(
				**get_regnet_args(depth=27, out_dim_slope=230.83, out_dim_base=640, log_step=2.53, dim_per_cardinal=373),
				attention=layers.SE,
				),
			params={
				'seer_ft_in1k_384': imagenet_params_config('regnety_256gf_seer_ft_in1k_384'),
				},
			),
		}
	return RegNet, configs
