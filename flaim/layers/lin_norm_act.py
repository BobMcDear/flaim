"""
Modules following a linear-normalization-activation structure:
- ConvBNAct: Convolution followed by batch normalization and an activation function.
- ConvLNAct: Convolution followed by layer normalization and an activation function.
"""


__all__ = [
	'ConvBNAct',
	'ConvLNAct',
	]


import typing as T

from flax import linen as nn

from .conv import Conv
from .identity import identity


class ConvBNAct(nn.Module):
	"""
	Convolution followed by batch normalization and an activation function.

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
		dilation (int): Dilation.
		Default is 1.
		bias (bool): Whether the convolution should have a bias term if bn is False.
		Default is True.
		bias_force (bool): Whether to force the convolution to have a bias term 
		even if bn is True.
		Default is False.
		bn (bool): Whether to have batch normalization after the convolution.
		If False, the training argument is ignored.
		Default is True.
		act (T.Callable): Activation function.
		Default is identity.
		tf (bool): Whether padding and batch_norm_eps should be fixed at 'same'
		and 1e-3 to conform to TensorFlow's settings.
		Default is False.
	"""
	out_dim: T.Optional[int] = None
	kernel_size: T.Union[T.Tuple[int, int], int] = 3
	stride: T.Union[T.Tuple[int, int], int] = 1
	padding: T.Optional[T.Union[str, int]] = None
	groups: T.Union[int, str] = 1
	dilation: int = 1
	bias: bool = True
	bias_force: bool = False
	bn: bool = True
	act: T.Callable = identity
	tf: bool = False

	@nn.compact
	def __call__(self, input, training: bool = True):
		output = Conv(
			out_dim=self.out_dim,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding='same' if self.tf else self.padding,
			groups=self.groups,
			dilation=self.dilation,
			bias=self.bias_force or (self.bias and not self.bn),
			)(input)
		output = nn.BatchNorm(
			use_running_average=not training,
			epsilon=1e-3 if self.tf else 1e-5,
			)(output) if self.bn else output
		output = self.act(output)
		return output


class ConvLNAct(nn.Module):
	"""
	Convolution followed by layer normalization and an activation function.

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
		dilation (int): Dilation.
		Default is 1.
		bias (bool): Whether to the convolution should have a bias term.
		Default is True.
		layer_norm_eps (T.Optional[float]): Epsilon value passed to layer normalization.
		If None, no normalization is applied.
		Default is 1e-6.
		act (T.Callable): Activation function.
		Default is identity.
	"""
	out_dim: T.Optional[int] = None
	kernel_size: T.Union[T.Tuple[int, int], int] = 3
	stride: T.Union[T.Tuple[int, int], int] = 1
	padding: T.Optional[T.Union[str, int]] = None
	groups: T.Union[int, str] = 1
	dilation: int = 1
	bias: bool = True
	layer_norm_eps: T.Optional[float] = 1e-6
	act: T.Callable = identity

	@nn.compact
	def __call__(self, input):
		output = Conv(
			out_dim=self.out_dim,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding,
			groups=self.groups,
			dilation=self.dilation,
			bias=self.bias,
			)(input)
		output = nn.LayerNorm(
			epsilon=self.layer_norm_eps,
			)(output) if self.layer_norm_eps else output
		output = self.act(output)
		return output
