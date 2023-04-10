"""
ResNet V2.
"""


import typing as T
from functools import partial

import jax
from flax import linen as nn

from .. import layers
from ..factory import inception_params_config, register_models
from .resnet import ResNet


def constant_padding(input, kernel_size: int = 3):
	"""
	Padding to maintain spatial dimensions for a given kernel size,
	similar to 'SAME' padding. Copied from the BiT repository.

	Args:
		input: Input to pad.
		kernel_size (int): Kernel size for which the input is to be padded.
		Default is 3.

	Returns: The input padded appropriately.
	"""
	pad_total = kernel_size - 1
	pad_beg = pad_total // 2
	pad_end = pad_total - pad_beg
	padded = jax.lax.pad(
		operand=input,
		padding_value=0.0,
		padding_config=(
			(0, 0, 0),
			(pad_beg, pad_end, 0),
			(pad_beg, pad_end, 0),
			(0, 0, 0)),
			)
	return padded


class BiTStem(nn.Module):
	"""
	ResNet stem with a weight-standardized convolution in lieu of batch
	normalization.

	Args:
		out_dim (int): Number of output channels.
		Default is 64.
		width_multiplier (int): Width multiplier.
		Default is 1.
		ws_eps (float): Epsilon value for weight standardization.
		Default is 1e-8.
	"""
	out_dim: int = 64
	width_multiplier: int = 1
	ws_eps: float = 1e-8

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.Conv(
			out_dim=self.width_multiplier*self.out_dim,
			kernel_size=7,
			stride=2,
			bias=False,
			ws_eps=self.ws_eps,
			)(input)
		output = constant_padding(output)
		output = layers.max_pool(
			input=output,
			stride=2,
			padding=0,
			)
		return output


class BiTHead(nn.Module):
	"""
	Head for BiT models, with group normalization and an activation function
	before the fully-connected layer.

	Args:
		n_classes (int): Number of output classes. If 0, the input is returned.
		If -1, all stages of the head, other than the final linear layer,
		are applied and the output returned.
		Default is 0.
		pool_fn (T.Callable): Pooling function.
		Default is global_avg_pool.
		group_norm_n_groups (int): Number of groups for group normalization.
		Default 32.
		bias (bool): Whether the linear layer should have
		a bias term.
		Default is True.
	"""
	n_classes: int = 0
	pool_fn: T.Callable = layers.global_avg_pool
	group_norm_n_groups: int = 32
	bias: bool = True

	@nn.compact
	def __call__(self, input):
		if self.n_classes == 0:
			return input

		output = nn.GroupNorm(
			num_groups=self.group_norm_n_groups,
			epsilon=1e-5,
			)(input)
		output = nn.relu(output)
		output = self.pool_fn(output, keep_axis=False)

		if self.n_classes != -1:
			output = nn.Dense(
				features=self.n_classes,
				use_bias=self.bias,
				)(output)

		return output


class BiTPreActBottleneckBlock(nn.Module):
	"""
	Pre-activation residual bottleneck block used by BiT, with group
	normalization and weight-standardized convolutions.

	Args:
		out_dim (int): Number of output channels.
		bottleneck_dim (int): Number of bottleneck channels.
		width_multiplier (int): Width multiplier.
		Default is 1.
		conv_block (T.Callable): Layer used as the
		middle group normalization-ReLU-convolution module. If None,
		it is set to layers.GNActConv.
		Default is None.
		cardinality (int): Cardinality.
		Default is 1.
		stride (int): Stride.
		Default is 1.
		ws_eps (float): Epsilon value for weight standardization.
		Default is 1e-8.
		group_norm_n_groups (int): Number of groups for group normalization.
		Default 32.
		downsample (T.Optional[T.Callable]): Downsampling module to use if input
		and output shapes are different. If None, a weight-standardized convolution
		is used.
		Default is None.
		attention (T.Callable): T.Callable returning an attention module
		applied to the data.
		Default is Identity.
		attention_pre (bool): Whether to apply the attention module before or after
		the second 1 x 1 convolution.
		Default is False.
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
	width_multiplier: int = 1
	conv_block: T.Optional[T.Callable] = None
	cardinality: int = 1
	stride: int = 1
	ws_eps: float = 1e-8
	group_norm_n_groups: int = 32
	downsample: T.Optional[T.Callable] = None
	attention: T.Callable = layers.Identity
	attention_pre: bool = False
	avg_downsample: T.Optional[str] = None

	@nn.compact
	def __call__(self, input, training: bool = True):
		out_dim = self.width_multiplier*self.out_dim
		bottleneck_dim = self.width_multiplier*self.bottleneck_dim

		residual = input

		input = nn.GroupNorm(
			num_groups=self.group_norm_n_groups,
			epsilon=1e-5,
			)(input)
		input = nn.relu(input)
		output = layers.Conv(
			out_dim=bottleneck_dim,
			kernel_size=1,
			bias=False,
			ws_eps=self.ws_eps,
			)(input)

		if self.avg_downsample == 'pre' and self.stride != 1:
			output = layers.avg_pool(
				input=output,
				stride=self.stride,
				)
		conv_block = self.conv_block or partial(
			layers.GNActConv,
			ws_eps=self.ws_eps,
			group_norm_n_groups=self.group_norm_n_groups,
			)
		output = conv_block(
			stride=1 if self.avg_downsample else self.stride,
			groups=self.cardinality,
			act=nn.relu,
			)(output)
		if self.avg_downsample == 'post' and self.stride != 1:
			output = layers.avg_pool(
				input=output,
				stride=self.stride,
				)

		if self.attention_pre:
			output = self.attention()(output)
		output = layers.GNActConv(
			out_dim=out_dim,
			kernel_size=1,
			ws_eps=self.ws_eps,
			group_norm_n_groups=self.group_norm_n_groups,
			act=nn.relu,
			)(output)
		if not self.attention_pre:
			output = self.attention()(output)

		if input.shape != output.shape:
			downsample = self.downsample or partial(
				layers.Conv,
				kernel_size=1,
				bias=False,
				ws_eps=self.ws_eps,
				)
			residual = downsample(
				out_dim=out_dim,
				stride=self.stride,
				)(input)

		output = residual+output
		return output


@register_models
def get_resnetv2_configs() -> T.Tuple[T.Type[ResNet], T.Dict]:
	"""
	Gets configurations for all available
	ResNet V2 models.

	Returns (T.Tuple[T.Type[ResNet], T.Dict]): The ResNet class and
	configurations of all available models.
	"""
	configs = {
		'resnetv2_bit_50x1': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				block=BiTPreActBottleneckBlock,
				conv_block=None,
				stem=BiTStem,
				downsample=None,
				head=BiTHead,
				),
			params={
				'dist_in1k_224': inception_params_config('resnetv2_bit_50x1_dist_in1k_224'),
				'in22k_224': inception_params_config('resnetv2_bit_50x1_in22k_224'),
				'in22k_ft_in1k_448': inception_params_config('resnetv2_bit_50x1_in22k_ft_in1k_448'),
				},
			),
		'resnetv2_bit_50x3': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				block=partial(BiTPreActBottleneckBlock, width_multiplier=3),
				conv_block=None,
				stem=partial(BiTStem, width_multiplier=3),
				downsample=None,
				head=BiTHead,
				),
			params={
				'in22k_224': inception_params_config('resnetv2_bit_50x3_in22k_224'),
				'in22k_ft_in1k_448': inception_params_config('resnetv2_bit_50x3_in22k_ft_in1k_448'),
				},
			),
		'resnetv2_bit_101x1': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				block=BiTPreActBottleneckBlock,
				conv_block=None,
				stem=BiTStem,
				downsample=None,
				head=BiTHead,
				),
			params={
				'in22k_224': inception_params_config('resnetv2_bit_101x1_in22k_224'),
				'in22k_ft_in1k_448': inception_params_config('resnetv2_bit_101x1_in22k_ft_in1k_448'),
				},
			),
		'resnetv2_bit_101x3': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				block=partial(BiTPreActBottleneckBlock, width_multiplier=3),
				conv_block=None,
				stem=partial(BiTStem, width_multiplier=3),
				downsample=None,
				head=BiTHead,
				),
			params={
				'in22k_224': inception_params_config('resnetv2_bit_101x3_in22k_224'),
				'in22k_ft_in1k_448': inception_params_config('resnetv2_bit_101x3_in22k_ft_in1k_448'),
				},
			),
		'resnetv2_bit_152x2': dict(
			model_args=dict(
				depths=(3, 8, 36, 3),
				block=partial(BiTPreActBottleneckBlock, width_multiplier=2),
				conv_block=None,
				stem=partial(BiTStem, width_multiplier=2),
				downsample=None,
				head=BiTHead,
				),
			params={
				'teacher_in22k_ft_in1k_224': inception_params_config('resnetv2_bit_152x2_teacher_in22k_ft_in1k_224'),
				'teacher_in22k_ft_in1k_384': inception_params_config('resnetv2_bit_152x2_teacher_in22k_ft_in1k_384'),
				'in22k_224': inception_params_config('resnetv2_bit_152x2_in22k_224'),
				'in22k_ft_in1k_448': inception_params_config('resnetv2_bit_152x2_in22k_ft_in1k_448'),
				},
			),
		}
	return ResNet, configs
