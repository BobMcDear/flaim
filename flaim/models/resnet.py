"""
ResNet.
"""


import typing as T
from functools import partial

from flax import linen as nn

from .. import layers
from ..factory import imagenet_params_config, register_models


class ResNetStem(nn.Module):
	"""
	ResNet stem.

	Args:
		out_dim (int): Number of output channels.
		Default is 64.
	"""
	out_dim: int = 64

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			kernel_size=7,
			stride=2,
			act=nn.relu,
			)(input, training=training)
		output = layers.max_pool(
			input=output,
			stride=2,
			)
		return output


class ResNetDStem(nn.Module):
	"""
	ResNet-D stem.

	Args:
		out_dims (T.Tuple[int, int, int]): Number of output
		channels of each 3 x 3 convolution.
		Default is (32, 32, 64).
		max_pool (bool): Whether to use max pool as the
		final downsampling module. If False, a 3 x 3 convolution
		is used.
		Default is True.
	"""
	out_dims: T.Tuple[int, int, int] = (32, 32, 64)
	max_pool: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.out_dims[0],
			stride=2,
			act=nn.relu,
			)(input, training=training)
		output = layers.ConvBNAct(
			out_dim=self.out_dims[1],
			act=nn.relu,
			)(output, training=training)
		output = layers.ConvBNAct(
			out_dim=self.out_dims[2],
			act=nn.relu,
			)(output, training=training)

		if self.max_pool:
			output = layers.max_pool(
				input=output,
				stride=2,
				)

		else:
			output = layers.ConvBNAct(
				out_dim=self.out_dims[-1],
				stride=2,
				act=nn.relu,
				)(output, training=training)

		return output


class ResNetDownsample(nn.Module):
	"""
	ResNet downsampling module.

	Args:
		out_dim (int): Number of output channels.
		stride (int): Stride, i.e., spatial downsampling
		factor.
		Default is 1.
		avg_pool (bool): Whether to average pool for downsampling
		instead of a strided convolution.
		Default is False.
	"""
	out_dim: int
	stride: int = 1
	avg_pool: bool = False

	@nn.compact
	def __call__(self, input, training: bool = True):
		if self.stride != 1 and self.avg_pool:
			input = layers.avg_pool(
				input=input,
				kernel_size=self.stride,
				stride=self.stride,
				padding='same',
				)
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			kernel_size=1,
			stride=1 if self.avg_pool else self.stride,
			)(input, training=training)
		return output


class BasicBlock(nn.Module):
	"""
	Residual basic block.

	Args:
		bottleneck_dim (int): Number of output channels of the first convolution.
		out_dim (T.Optional[int]): Number of output channels of the second convolution.
		If None, it is set to bottleneck_dim.
		Default is None.
		conv_block (T.Callable): Layer used as the
		first convolution-batch normalization-ReLU module.
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
		the second 3 x 3 convolution.
		Default is False.
		avg_downsample (T.Optional[str]): If 'pre', average pooling is used
		before the first 3 x 3 convolution for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If 'post', average pooling is used
		after the first 3 x 3 convolution for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If None, spatial downsampling
		is performed by the first 3 x 3 convolution.
		Default is None.
	"""
	bottleneck_dim: int
	out_dim: T.Optional[int] = None
	conv_block: T.Callable = layers.ConvBNAct
	cardinality: int = 1
	stride: int = 1
	downsample: T.Callable = ResNetDownsample
	attention: T.Callable = layers.Identity
	attention_pre: bool = False
	avg_downsample: T.Optional[str] = None

	@nn.compact
	def __call__(self, input, training: bool = True):
		if self.avg_downsample == 'pre' and self.stride != 1:
			input = layers.avg_pool(
				input=input,
				stride=self.stride,
				)
		output = self.conv_block(
			out_dim=self.bottleneck_dim,
			stride=1 if self.avg_downsample else self.stride,
			groups=self.cardinality,
			act=nn.relu,
			)(input, training=training)
		if self.avg_downsample == 'post' and self.stride != 1:
			output = layers.avg_pool(
				input=output,
				stride=self.stride,
				)

		if self.attention_pre:
			output = self.attention()(output)
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			groups=self.cardinality,
			)(output, training=training)
		if not self.attention_pre:
			output = self.attention()(output)

		if input.shape != output.shape:
			input = self.downsample(
				out_dim=self.out_dim or self.bottleneck_dim,
				stride=self.stride,
				)(input, training=training)

		output = input+output
		output = nn.relu(output)
		return output


class BottleneckBlock(nn.Module):
	"""
	Residual bottleneck block.

	Args:
		out_dim (int): Number of output channels.
		bottleneck_dim (int): Number of bottleneck channels.
		conv_block (T.Callable): Layer used as the
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
	conv_block: T.Callable = layers.ConvBNAct
	cardinality: int = 1
	stride: int = 1
	downsample: T.Callable = ResNetDownsample
	attention: T.Callable = layers.Identity
	attention_pre: bool = False
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
		output = self.conv_block(
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
			output = self.attention()(output)
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			kernel_size=1,
			)(output, training=training)
		if not self.attention_pre:
			output = self.attention()(output)

		if input.shape != output.shape:
			input = self.downsample(
				out_dim=self.out_dim,
				stride=self.stride,
				)(input, training=training)

		output = input+output
		output = nn.relu(output)

		return output


class ResNetStage(nn.Module):
	"""
	ResNet stage.

	Args:
		depth (int): Depth.
		out_dim (int): Number of output channels.
		bottleneck_dim (int): Number of bottleneck channels.
		block (T.Callable): Block used to construct the stage.
		Default is BottleneckBlock.
		conv_block (T.Callable): Layer used as the
		first convolution-batch normalization-ReLU module
		of basic blocks or the middle convolution-batch normalization-ReLU
		module of bottleneck blocks.
		Default is layers.ConvBNAct.
		cardinality (int): Cardinality.
		Default is 1.
		stride (int): Stride.
		Default is 1.
		downsample (T.Callable): Downsampling module to use if input and output
		shapes are different.
		Default is ResNetDownsample.
		attention (T.Callable): T.Callable returning an attention module
		applied to the data.
		Default is Identity.
		attention_pre (bool): Whether to apply the attention module before or after
		the second 3 x 3 convolution in basic blocks or the second 1 x 1 convolution
		in bottleneck blocks.
		Default is False.
		avg_downsample (T.Optional[str]): If 'pre', average pooling is used
		before the first 3 x 3 convolution of each block for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If 'post', average pooling is used
		after the first 3 x 3 convolution of each block for spatial downsampling if stride is not 1,
		and the convolution's stride is set to 1. If None, spatial downsampling
		is performed by the 3 x 3 convolutions of each block.
		Default is None.
	"""
	depth: int
	out_dim: int
	bottleneck_dim: int
	block: T.Callable = BottleneckBlock
	conv_block: T.Callable = layers.ConvBNAct
	cardinality: int = 1
	stride: int = 1
	downsample: T.Callable = ResNetDownsample
	attention: T.Callable = layers.Identity
	attention_pre: bool = False
	avg_downsample: T.Optional[str] = None

	@nn.compact
	def __call__(self, input, training: bool = True):
		for block_ind in range(self.depth):
			input = self.block(
				out_dim=self.out_dim,
				bottleneck_dim=self.bottleneck_dim,
				conv_block=self.conv_block,
				cardinality=self.cardinality,
				stride=self.stride if block_ind == 0 else 1,
				downsample=self.downsample,
				attention=self.attention,
				attention_pre=self.attention_pre,
				avg_downsample=self.avg_downsample,
				)(input, training=training)
		return input


class ResNet(nn.Module):
	"""
	ResNet, with support for different bottleneck dimensions, stems,
	etc.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		block (T.Callable): Block used to construct the network.
		Default is BottleneckBlock.
		conv_block (T.Callable): Layer used as the
		first convolution-batch normalization-ReLU module
		of basic blocks or the middle convolution-batch normalization-ReLU
		module of bottleneck blocks.
		Default is layers.ConvBNAct.
		cardinality (int): Cardinality.
		Default is 1.
		dim_per_cardinal_first_stage (int): Number of channels per cardinal group for the
		first stage.
		Default is 64.
		stem (T.Callable): Stem.
		Default is ResNetStem.
		downsample (T.Callable): Downsampling module to use if input and output
		shapes are different.
		Default is ResNetDownsample.
		attention (T.Callable): T.Callable returning an attention module
		applied to the data.
		Default is Identity.
		attention_pre (bool): Whether to apply the attention module before or after
		the second 3 x 3 convolution in basic blocks or the second 1 x 1 convolution
		in bottleneck blocks.
		Default is False.
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
	block: T.Callable = BottleneckBlock
	conv_block: T.Callable = layers.ConvBNAct
	cardinality: int = 1
	dim_per_cardinal_first_stage: int = 64
	stem: T.Callable = ResNetStem
	downsample: T.Callable = ResNetDownsample
	attention: T.Callable = layers.Identity
	attention_pre: bool = False
	avg_downsample: T.Optional[str] = None
	head: T.Type = layers.Head
	n_classes: int = 0

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = self.stem()(input, training=training)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		for stage_ind, depth in enumerate(self.depths):
			bottleneck_dim = self.cardinality * self.dim_per_cardinal_first_stage * (2**stage_ind)
			output = ResNetStage(
				depth=depth,
				out_dim=bottleneck_dim if self.block == BasicBlock else 2 ** (stage_ind+8),
				bottleneck_dim=bottleneck_dim,
				block=self.block,
				conv_block=self.conv_block,
				cardinality=self.cardinality,
				stride=1 if stage_ind == 0 else 2,
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

		output = self.head(
			n_classes=self.n_classes,
			)(output)

		return output


@register_models
def get_resnet_configs() -> T.Tuple[T.Type[ResNet], T.Dict]:
	"""
	Gets configurations for all available
	ResNet models.

	Returns (T.Tuple[T.Type[ResNet], T.Dict]): The ResNet class and
	configurations of all available models.
	"""
	configs = {
		'resnet10t': dict(
			model_args=dict(
				depths=(1, 1, 1, 1),
				block=BasicBlock,
				stem=partial(ResNetDStem, out_dims=(24, 32, 64)),
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_176': imagenet_params_config('resnet10t'),
				},
			),
		'resnet14t': dict(
			model_args=dict(
				depths=(1, 1, 1, 1),
				stem=partial(ResNetDStem, out_dims=(24, 32, 64)),
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_176': imagenet_params_config('resnet14t'),
				},
			),
		'resnet18': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				block=BasicBlock,
			),
			params={
				'in1k_224': imagenet_params_config('resnet18'),
				'ssl_ft_in1k_224': imagenet_params_config('resnet18_ssl'),
				'swsl_ft_in1k_224': imagenet_params_config('resnet18_swsl'),
				},
			),
		'resnet18d': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				block=BasicBlock,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_224': imagenet_params_config('resnet18d'),
				},
			),
		'resnet26': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
			),
			params={
				'in1k_224': imagenet_params_config('resnet26'),
				},
			),
		'resnet26d': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_224': imagenet_params_config('resnet26d'),
				},
			),
		'resnet26t': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				stem=partial(ResNetDStem, out_dims=(24, 32, 64)),
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_256': imagenet_params_config('resnet26t'),
				},
			),
		'resnet34': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				block=BasicBlock,
			),
			params={
				'in1k_224': imagenet_params_config('resnet34'),
				},
			),
		'resnet34d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				block=BasicBlock,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_224': imagenet_params_config('resnet34d'),
				},
			),
		'resnet50': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
			),
			params={
				'in1k_224': imagenet_params_config('resnet50'),
				'ssl_ft_in1k_224': imagenet_params_config('resnet50_ssl'),
				'swsl_ft_in1k_224': imagenet_params_config('resnet50_swsl'),
				},
			),
		'resnet50d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_224': imagenet_params_config('resnet50d'),
				},
			),
		'resnet101': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
			),
			params={
				'in1k_224': imagenet_params_config('resnet101'),
				},
			),
		'resnet101d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_256': imagenet_params_config('resnet101d'),
				},
			),
		'resnet152': dict(
			model_args=dict(
				depths=(3, 8, 36, 3),
			),
			params={
				'in1k_224': imagenet_params_config('resnet152'),
				},
			),
		'resnet152d': dict(
			model_args=dict(
				depths=(3, 8, 36, 3),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_256': imagenet_params_config('resnet152d'),
				},
			),
		'resnet200d': dict(
			model_args=dict(
				depths=(3, 24, 36, 3),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
			),
			params={
				'in1k_256': imagenet_params_config('resnet200d'),
				},
			),
		'wide_resnet50_2': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				dim_per_cardinal_first_stage=128,
			),
			params={
				'in1k_224': imagenet_params_config('wide_resnet50_2'),
				},
			),
		'wide_resnet101_2': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				dim_per_cardinal_first_stage=128,
			),
			params={
				'in1k_224': imagenet_params_config('wide_resnet101_2'),
				},
			),
		'resnext50_32x4d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=4,
			),
			params={
				'in1k_224': imagenet_params_config('resnext50_32x4d'),
				'ssl_ft_in1k_224': imagenet_params_config('resnext50_32x4d_ssl'),
				'swsl_ft_in1k_224': imagenet_params_config('resnext50_32x4d_swsl'),
				},
			),
		'resnext101_32x4d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=4,
			),
			params={
				'ssl_ft_in1k_224': imagenet_params_config('resnext101_32x4d_ssl'),
				'swsl_ft_in1k_224': imagenet_params_config('resnext101_32x4d_swsl'),
				},
			),
		'resnext101_32x8d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=8,
			),
			params={
				'in1k_224': imagenet_params_config('resnext101_32x8d'),
				'ssl_ft_in1k_224': imagenet_params_config('resnext101_32x8d_ssl'),
				'wsl_ft_in1k_224': imagenet_params_config('resnext101_32x8d_wsl'),
				'swsl_ft_in1k_224': imagenet_params_config('resnext101_32x8d_swsl'),
				},
			),
		'resnext101_32x16d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=16,
			),
			params={
				'ssl_ft_in1k_224': imagenet_params_config('resnext101_32x16d_ssl'),
				'wsl_ft_in1k_224': imagenet_params_config('resnext101_32x16d_wsl'),
				'swsl_ft_in1k_224': imagenet_params_config('resnext101_32x16d_swsl'),
				},
			),
		'resnext101_32x32d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=32,
			),
			params={
				'wsl_ft_in1k_224': imagenet_params_config('resnext101_32x32d_wsl'),
				},
			),
		'resnext101_32x48d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=48,
			),
			params={
				'wsl_ft_in1k_224': imagenet_params_config('resnext101_32x48d_wsl'),
				},
			),
		'resnext101_64x4d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=64,
				dim_per_cardinal_first_stage=4,
			),
			params={
				'in1k_224': imagenet_params_config('resnext101_64x4d'),
				},
			),
		'seresnet50': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				attention=layers.SE,
			),
			params={
				'in1k_224': imagenet_params_config('seresnet50'),
				},
			),
		'seresnet152d': dict(
			model_args=dict(
				depths=(3, 8, 36, 3),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.SE,
			),
			params={
				'in1k_256': imagenet_params_config('seresnet152d'),
				},
			),
		'seresnext26d_32x4d': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				cardinality=32,
				dim_per_cardinal_first_stage=4,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.SE,
			),
			params={
				'in1k_224': imagenet_params_config('seresnext26d_32x4d'),
				},
			),
		'seresnext26t_32x4d': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				cardinality=32,
				dim_per_cardinal_first_stage=4,
				stem=partial(ResNetDStem, out_dims=(24, 32, 64)),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.SE,
			),
			params={
				'in1k_224': imagenet_params_config('seresnext26t_32x4d'),
				},
			),
		'seresnext50_32x4d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=4,
				attention=layers.SE,
			),
			params={
				'in1k_224': imagenet_params_config('seresnext50_32x4d'),
				},
			),
		'seresnext101_32x8d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=8,
				attention=layers.SE,
			),
			params={
				'in1k_224': imagenet_params_config('seresnext101_32x8d'),
				},
			),
		'seresnext101d_32x8d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				cardinality=32,
				dim_per_cardinal_first_stage=8,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.SE,
			),
			params={
				'in1k_224': imagenet_params_config('seresnext101d_32x8d'),
				},
			),
		'ecaresnet26t': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				stem=partial(ResNetDStem, out_dims=(24, 32, 64)),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.ECA,
			),
			params={
				'in1k_256': imagenet_params_config('ecaresnet26t'),
				},
			),
		'ecaresnet50_light': dict(
			model_args=dict(
				depths=(1, 1, 11, 3),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.ECA,
			),
			params={
				'in1k_224': imagenet_params_config('ecaresnet50_light'),
				},
			),
		'ecaresnet50d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.ECA,
			),
			params={
				'in1k_224': imagenet_params_config('ecaresnet50d'),
				},
			),
		'ecaresnet50t': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				stem=partial(ResNetDStem, out_dims=(24, 32, 64)),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.ECA,
			),
			params={
				'in1k_256': imagenet_params_config('ecaresnet50t'),
				},
			),
		'ecaresnet101d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.ECA,
			),
			params={
				'in1k_224': imagenet_params_config('ecaresnet101d'),
				},
			),
		'ecaresnet269d': dict(
			model_args=dict(
				depths=(3, 30, 48, 8),
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=layers.ECA,
			),
			params={
				'in1k_320': imagenet_params_config('ecaresnet269d'),
				},
			),
		'resnetrs50': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				stem=partial(ResNetDStem, max_pool=False),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=partial(layers.SE, reduction_factor=4),
			),
			params={
				'in1k_160': imagenet_params_config('resnetrs50'),
				},
			),
		'resnetrs101': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				stem=partial(ResNetDStem, max_pool=False),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=partial(layers.SE, reduction_factor=4),
			),
			params={
				'in1k_192': imagenet_params_config('resnetrs101'),
				},
			),
		'resnetrs152': dict(
			model_args=dict(
				depths=(3, 8, 36, 3),
				stem=partial(ResNetDStem, max_pool=False),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=partial(layers.SE, reduction_factor=4),
			),
			params={
				'in1k_256': imagenet_params_config('resnetrs152'),
				},
			),
		'resnetrs200': dict(
			model_args=dict(
				depths=(3, 24, 36, 3),
				stem=partial(ResNetDStem, max_pool=False),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=partial(layers.SE, reduction_factor=4),
			),
			params={
				'in1k_256': imagenet_params_config('resnetrs200'),
				},
			),
		'resnetrs270': dict(
			model_args=dict(
				depths=(4, 29, 53, 4),
				stem=partial(ResNetDStem, max_pool=False),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=partial(layers.SE, reduction_factor=4),
			),
			params={
				'in1k_256': imagenet_params_config('resnetrs270'),
				},
			),
		'resnetrs350': dict(
			model_args=dict(
				depths=(4, 36, 72, 4),
				stem=partial(ResNetDStem, max_pool=False),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=partial(layers.SE, reduction_factor=4),
			),
			params={
				'in1k_288': imagenet_params_config('resnetrs350'),
				},
			),
		'resnetrs420': dict(
			model_args=dict(
				depths=(4, 44, 87, 4),
				stem=partial(ResNetDStem, max_pool=False),
				downsample=partial(ResNetDownsample, avg_pool=True),
				attention=partial(layers.SE, reduction_factor=4),
			),
			params={
				'in1k_320': imagenet_params_config('resnetrs420'),
				},
			),
		'skresnet18': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				block=BasicBlock,
				conv_block=layers.SK,
			),
			params={
				'in1k_224': imagenet_params_config('skresnet18'),
				},
			),
		'skresnet34': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				block=BasicBlock,
				conv_block=layers.SK,
			),
			params={
				'in1k_224': imagenet_params_config('skresnet34'),
				},
			),
		'skresnext50_32x4d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				conv_block=partial(layers.SK, reduction_factor=16, min_reduction_dim=32, split=False),
				cardinality=32,
				dim_per_cardinal_first_stage=4,
			),
			params={
				'in1k_224': imagenet_params_config('skresnext50_32x4d'),
				},
			),
		'resnest14_2s1x64d': dict(
			model_args=dict(
				depths=(1, 1, 1, 1),
				conv_block=layers.SplAt,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='post',
			),
			params={
				'in1k_224': imagenet_params_config('resnest14_2s1x64d'),
				},
			),
		'resnest26_2s1x64d': dict(
			model_args=dict(
				depths=(2, 2, 2, 2),
				conv_block=layers.SplAt,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='post',
			),
			params={
				'in1k_224': imagenet_params_config('resnest26_2s1x64d'),
				},
			),
		'resnest50_1s4x24d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				conv_block=partial(layers.SplAt, radix=1),
				cardinality=4,
				dim_per_cardinal_first_stage=24,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='pre',
			),
			params={
				'in1k_224': imagenet_params_config('resnest50_1s4x24d_in1k_224'),
				},
			),
		'resnest50_2s1x64d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				conv_block=layers.SplAt,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='post',
			),
			params={
				'in1k_224': imagenet_params_config('resnest50_2s1x64d'),
				},
			),
		'resnest50_4s2x40d': dict(
			model_args=dict(
				depths=(3, 4, 6, 3),
				conv_block=partial(layers.SplAt, radix=4),
				cardinality=2,
				dim_per_cardinal_first_stage=40,
				stem=ResNetDStem,
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='pre',
			),
			params={
				'in1k_224': imagenet_params_config('resnest50_4s2x40d_in1k_224'),
				},
			),
		'resnest101_2s1x64d': dict(
			model_args=dict(
				depths=(3, 4, 23, 3),
				conv_block=layers.SplAt,
				stem=partial(ResNetDStem, out_dims=(64, 64, 128)),
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='post',
			),
			params={
				'in1k_256': imagenet_params_config('resnest101_2s1x64d'),
				},
			),
		'resnest200_2s1x64d': dict(
			model_args=dict(
				depths=(3, 24, 36, 3),
				conv_block=layers.SplAt,
				stem=partial(ResNetDStem, out_dims=(64, 64, 128)),
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='post',
			),
			params={
				'in1k_320': imagenet_params_config('resnest200_2s1x64d'),
				},
			),
		'resnest269_2s1x64d': dict(
			model_args=dict(
				depths=(3, 30, 48, 8),
				conv_block=layers.SplAt,
				stem=partial(ResNetDStem, out_dims=(64, 64, 128)),
				downsample=partial(ResNetDownsample, avg_pool=True),
				avg_downsample='post',
			),
			params={
				'in1k_416': imagenet_params_config('resnest269_2s1x64d'),
				},
			),
		}
	return ResNet, configs
