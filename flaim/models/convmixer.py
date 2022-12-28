"""
ConvMixer.
"""


import typing as T

from flax import linen as nn

from .. import layers
from .factory import register_configs


class ConvActBN(nn.Module):
	"""
	Convolution followed by an activation function and batch normalization.

	Args:
		out_dim (T.Optional[int]): Number of output channels.
		If None, it is set to the number of input channels.
		Default is None.
		kernel_size (T.Union[T.Tuple[int, int], int]): Kernel size.
		If an int, this value is used along both spatial dimensions.
		stride (T.Union[T.Tuple[int, int], int]): Stride. If an int,
		this value is used along both spatial dimensions.
		Default is 1.
		padding (T.Optional[T.Union[str, int]]): Padding. If None,
		it is set so the spatial dimensions are exactly divided by stride.
		If an int, this value is used along both spatial dimensions.
		Default is None.
		groups (T.Union[int, str]): Number of groups. If 'dw', a depthwise convolution is performed.
		Default is 1.
		act (T.Callable): Activation function.
		Default is identity.
	"""
	out_dim: T.Optional[int] = None
	kernel_size: T.Union[T.Tuple[int, int], int] = 3
	stride: T.Union[T.Tuple[int, int], int] = 1
	padding: T.Optional[T.Union[str, int]] = None
	groups: T.Union[int, str] = 1
	act: T.Callable = layers.identity

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = layers.Conv(
			out_dim=self.out_dim,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding,
			groups=self.groups,
			)(input)
		output = self.act(output)
		output = nn.BatchNorm(
			use_running_average=not training,
			)(output)
		return output


class ConvMixerBlock(nn.Module):
	"""
	ConvMixer block.

	Args:
		kernel_size (int): Kernel size.
		Default is 9.
		act (T.Callable): Activation function.
		Default is layers.gelu.
	"""
	kernel_size: int = 9
	act: T.Callable = layers.gelu

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = input + ConvActBN(
				kernel_size=self.kernel_size,
				padding='same',
				groups='dw',
				act=self.act,
				)(input, training=training)
		output = ConvActBN(
				kernel_size=1,
				act=self.act,
				)(output, training=training)
		return output


class ConvMixer(nn.Module):
	"""
	ConvMixer.

	Args:
		depth (int): Depth.
		dim (int): Dimension throughout the network.
		patch_size (int): Patch size.
		Default is 7.
		kernel_size (int): Kernel size.
		Default is 9.
		act (T.Callable): Activation function.
		Default is layers.gelu.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depth: int
	dim: int
	patch_size: int = 7
	kernel_size: int = 9
	act: T.Callable = layers.gelu
	n_classes: int = 0

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = ConvActBN(
			out_dim=self.dim,
			kernel_size=self.patch_size,
			stride=self.patch_size,
			padding=0,
			act=self.act,
			)(input, training=training)
		self.sow(
			col='intermediates',
			name='block_0',
			value=output,
			)
		
		for block_ind in range(self.depth):
			output = ConvMixerBlock(
				kernel_size=self.kernel_size,
				act=self.act,
				)(output, training=training)
			self.sow(
				col='intermediates',
				name=f'block_{block_ind+1}',
				value=output,
				)

		output = layers.Head(
			n_classes=self.n_classes,
			)(output)
		
		return output


@register_configs
def get_convmixer_configs() -> T.Tuple[T.Type[ConvMixer], T.Dict]:
	"""
	Gets configurations for all available
	ConvMixer models.

	Returns (T.Tuple[T.Type[ConvMixer], T.Dict]): The ConvMixer class and
	configurations of all available models.
	"""
	configs = {
		'convmixer20_1024d_patch14_kernel9': {
			'depth': 20,
			'dim': 1024,
			'patch_size': 14,
			},
		'convmixer20_1536d_patch7_kernel9': {
			'depth': 20,
			'dim': 1536,
			},
		'convmixer32_768d_patch7_kernel7': {
			'depth': 32,
			'dim': 768,
			'kernel_size': 7,
			'act': nn.relu,
			},
		}
	return ConvMixer, configs
