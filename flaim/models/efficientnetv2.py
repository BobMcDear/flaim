"""
EfficientNetV2.
"""


import typing as T

from flax import linen as nn

from .. import layers
from .factory import NORM_STATS, register_configs


class ResConvBNAct(nn.Module):
	"""
	Convolution followed by batch normalization and an activation function,
	plus a residual connection.

	Args:
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
		expansion_factor (int): Expansion factor for the
		output dimension.
		Defautl is 1.
		act (T.Callable): Activation function.
		Default is nn.silu.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
	"""
	out_dim: int
	stride: int = 1
	expansion_factor: int = 1
	act: T.Callable = nn.silu
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.expansion_factor*self.out_dim,
			stride=self.stride,
			act=self.act,
			tf=self.tf,
			)(input, training=training)
		return input+output if input.shape == output.shape else output


class FusedMBConv(nn.Module):
	"""
	Fused MBConv.

	Args:
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
		expansion_factor (int): Expansion factor for the
		width.
		Default is 1.
		act (T.Callable): Activation function.
		Default is nn.silu.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
	"""
	out_dim: int
	stride: int = 1
	expansion_factor: int = 1
	act: T.Callable = nn.silu
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.expansion_factor*input.shape[-1],
			stride=self.stride,
			act=self.act,
			tf=self.tf,
			)(input, training=training)
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			kernel_size=1,
			tf=self.tf,
			)(output, training=training)
		return input+output if input.shape == output.shape else output


class DWSeparableConv(nn.Module):
	"""
	Depthwise separable convolution.

	Args:
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
		act (T.Callable): Activation function.
		Default is nn.silu.
		res (bool): Whether to apply a residual summation
		if the input and output shapes are identical.
		Default is False.
		se_reduction_factor (T.Optional[int]): Reduction factor for
		squeeze-and-excitation. If None, no squeeze-and-excitation
		is applied.
		Default is 4.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
	"""
	out_dim: int
	stride: int = 1
	act: T.Callable = nn.silu
	res: bool = False
	se_reduction_factor: T.Optional[int] = 4
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			stride=self.stride,
			groups='dw',
			act=self.act,
			tf=self.tf,
			)(input, training=training)
		output = layers.SE(
			reduction_factor=self.se_reduction_factor,
			act=self.act,
			)(output) if self.se_reduction_factor else output
		output = layers.ConvBNAct(
			out_dim=self.out_dim,
			kernel_size=1,
			tf=self.tf,
			)(output, training=training)
		return input+output if self.res and input.shape == output.shape else output


class MBConv(nn.Module):
	"""
	MBConv, a.k.a. inverted residual block.

	Args:
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
		expansion_factor (int): Expansion factor for
		the inverted bottleneck.
		Default is 1.
		act (T.Callable): Activation function.
		Default is nn.silu.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
	"""
	out_dim: int
	stride: int = 1
	expansion_factor: int = 1
	act: T.Callable = nn.silu
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.expansion_factor*input.shape[-1],
			kernel_size=1,
			act=self.act,
			tf=self.tf,
			)(input, training=training)
		output = DWSeparableConv(
			out_dim=self.out_dim,
			stride=self.stride,
			act=self.act,
			se_reduction_factor=self.expansion_factor*4,
			tf=self.tf,
			)(output, training=training)
		return input+output if input.shape == output.shape else output


class EfficientNetV2Stage(nn.Module):
	"""
	EfficientNetV2 stage.

	Args:
		block (T.Callable): Block used to construct the stage.
		depth (int): Depth.
		out_dim (int): Number of output channels.
		stride (int): Stride.
		Default is 1.
		expansion_factor (int): Expansion factor.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
	"""
	block: T.Callable
	depth: int
	out_dim: int
	stride: int = 1
	expansion_factor: int = 1
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		for block_ind in range(self.depth):
			input = self.block(
				out_dim=self.out_dim,
				stride=self.stride if block_ind == 0 else 1,
				expansion_factor=self.expansion_factor,
				tf=self.tf,
				)(input, training=training)
		return input


class EfficientNetV2Head(nn.Module):
	"""
	EfficientNetV2 head.

	Args:
		n_classes (int): Number of output classes. If 0, the input is returned.
		If -1, all stages of the head, other than the final linear layer,
		are applied and the output returned. 
		Default is 0.
		conv_bn_act_dim (int): Number of output channels of
		the pre-pooling convolution-normalization-activation block.
		Default is 1280.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
	"""
	n_classes: int = 0
	conv_bn_act_dim: int = 1280
	tf: bool = True

	@nn.compact
	def __call__(self, input, training: bool = True):
		if self.n_classes == 0:
			return input
		
		output = layers.ConvBNAct(
			out_dim=self.conv_bn_act_dim,
			kernel_size=1,
			act=nn.silu,
			tf=self.tf,
			)(input, training=training)
		output = layers.Head(
			n_classes=self.n_classes,
			)(output)
			
		return output


class EfficientNetV2(nn.Module):
	"""
	EfficientNetV2.

	Args:
		blocks (T.Tuple[T.Callable, ...]): Block used to construct each stage.
		depths (T.Tuple[int, ...]): Depth of each stage.
		out_dims (T.Tuple[int, ...]): Number of output channels of each stage.
		strides (T.Tuple[int, ...]): Stride of each stage.
		expansion_factors (T.Tuple[int, ...]): Expansion factor of
		each stage.
		tf (bool): Whether to use batch normalization epsilon of
		1e-3 and padding of 'same' for compatibility with TensorFlow.
		Default is True.
		n_classes (int): Number of output classes. If 0, there is no head,
		head_bottleneck_dim is ignored, and the raw final features are returned.
		If -1, all stages of the head, other than the final linear layer, are applied
		and the output returned.
		Default is 0.
		head_conv_bn_act_dim (int): Number of output channels of
		the pre-pooling convolution-normalization-activation block.
		Default is 1280.
	"""
	blocks: T.Tuple[T.Callable, ...]
	depths: T.Tuple[int, ...]
	out_dims: T.Tuple[int, ...]
	strides: T.Tuple[int, ...]
	expansion_factors: T.Tuple[int, ...]
	tf: bool = True
	n_classes: int = 0
	head_conv_bn_act_dim: int = 1280 

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.ConvBNAct(
			out_dim=self.out_dims[0],
			stride=2,
			act=nn.silu,
			tf=self.tf,
			)(input, training=training)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)

		for stage_ind in range(len(self.depths)):
			output = EfficientNetV2Stage(
				block=self.blocks[stage_ind],
				depth=self.depths[stage_ind],
				out_dim=self.out_dims[stage_ind],
				stride=self.strides[stage_ind],
				expansion_factor=self.expansion_factors[stage_ind],
				tf=self.tf,
				)(output, training=training)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output,
				)
		
		output = EfficientNetV2Head(
			n_classes=self.n_classes,
			conv_bn_act_dim=self.head_conv_bn_act_dim,
			tf=self.tf,
			)(output, training=training)
		
		return output


@register_configs
def get_efficientnetv2_configs() -> T.Tuple[T.Type[EfficientNetV2], T.Dict]:
	"""
	Gets configurations for all available
	EfficientNetV2 models.

	Returns (T.Tuple[T.Type[EfficientNetV2], T.Dict]): The EfficientNetV2 class and
	configurations of all available models.
	"""
	configs = {
		'efficientnetv2_small': {
			'depths': (2, 4, 4, 6, 9, 15),
			'out_dims': (24, 48, 64, 128, 160, 256),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2),
			'expansion_factors': (1, 4, 4, 4, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_medium': {
			'depths': (3, 5, 5, 7, 14, 18, 5),
			'out_dims': (24, 48, 80, 160, 176, 304, 512),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_large': {
			'depths': (4, 7, 7, 10, 19, 25, 7),
			'out_dims': (32, 64, 96, 192, 224, 384, 640),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_small_in22k': {
			'depths': (2, 4, 4, 6, 9, 15),
			'out_dims': (24, 48, 64, 128, 160, 256),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_medium_in22k': {
			'depths': (3, 5, 5, 7, 14, 18, 5),
			'out_dims': (24, 48, 80, 160, 176, 304, 512),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_large_in22k': {
			'depths': (4, 7, 7, 10, 19, 25, 7),
			'out_dims': (32, 64, 96, 192, 224, 384, 640),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_xlarge_in22k': {
			'depths': (4, 8, 8, 16, 24, 32, 8),
			'out_dims': (32, 64, 96, 192, 256, 512, 640),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_small_in22ft1k': {
			'depths': (2, 4, 4, 6, 9, 15),
			'out_dims': (24, 48, 64, 128, 160, 256),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_medium_in22ft1k': {
			'depths': (3, 5, 5, 7, 14, 18, 5),
			'out_dims': (24, 48, 80, 160, 176, 304, 512),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_large_in22ft1k': {
			'depths': (4, 7, 7, 10, 19, 25, 7),
			'out_dims': (32, 64, 96, 192, 224, 384, 640),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		'efficientnetv2_xlarge_in22ft1k': {
			'depths': (4, 8, 8, 16, 24, 32, 8),
			'out_dims': (32, 64, 96, 192, 256, 512, 640),
			'blocks': (ResConvBNAct, FusedMBConv, FusedMBConv, MBConv, MBConv, MBConv, MBConv),
			'strides': (1, 2, 2, 2, 1, 2, 1),
			'expansion_factors': (1, 4, 4, 4, 6, 6, 6),
			'norm_stats': NORM_STATS['inception'],
			},
		}
	return EfficientNetV2, configs
