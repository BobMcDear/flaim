"""
MLP-like layers:
- MLP: Multilayer perceptron with one hidden layer.
- ConvMLP: Multilayer perceptron with one hidden layer, implemented using convolutions.
- TransformerMLP: Transformer-style MLP by Vaswani et al.

References:
- Vaswani et al. Attention Is All You Need.
"""


__all__ = [
	'MLP',
	'ConvMLP',
	'TransformerMLP',
	]


import typing as T
from math import sqrt

from flax import linen as nn
from jax import numpy as jnp

from .acts import gelu
from .layer_scale import LayerScale
from .lin_norm_act import Conv


def get_dims(
	in_dim: int,
	out_dim: T.Optional[int],
	hidden_dim_expansion_factor: T.Optional[float],
	hidden_dim: T.Optional[int],
	) -> T.Tuple[int, int]:
	"""
	Gets hidden and output dimensions for an MLP.

	Args:
		in_dim (int): Number of input neurons.
		out_dim (T.Optional[int]): Number of output neurons. If None,
		it is set to the number of input channels.
		Default is None.
		hidden_dim_expansion_factor (T.Optional[float]): Factor of expansion for the 
		hidden layer. If None, hidden_dim must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
		hidden_dim (T.Optional[int]): Number of neurons in the hidden layer.
		If None, hidden_dim_expansion_factor must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
	
	Returns (T.Tuple[int, int]): Hidden and output dimensions.
	"""
	return (
		hidden_dim or int(hidden_dim_expansion_factor*in_dim),
		out_dim or in_dim,
		)


class MLP(nn.Module):
	"""
	Multilayer perceptron with one hidden layer.

	Args:
		out_dim (T.Optional[int]): Number of output neurons. If None,
		it is set to the number of input channels.
		Default is None.
		hidden_dim_expansion_factor (T.Optional[float]): Factor of expansion for the 
		hidden layer. If None, hidden_dim must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
		hidden_dim (T.Optional[int]): Number of neurons in the hidden layer.
		If None, hidden_dim_expansion_factor must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
		act (T.Callable): Activation for the hidden layer.
		Default is nn.relu.
		output_act (bool): Whether to apply activation on the output.
		Default is False. 
		bias (bool): Whether the linear layers should have a bias term. If bn is True,
		the first linear layer would not have a bias term.
		Default is True.
		bias_force (bool): Whether to force the first linear layer to have a bias term 
		even if bn is True.
		Default is False.
		bn (bool): Whether to apply batch normalization before the activation.
		If False, the training argument is ignored.
		Default is False.
		dw_kernel_size (T.Optional[int]): Kernel size of a depthwise convolution
		applied immediately after the first linear layer. If None,
		no depthwise convolution is applied.
		Default is None.
	"""
	out_dim: T.Optional[int] = None
	hidden_dim_expansion_factor: T.Optional[float] = None
	hidden_dim: T.Optional[int] = None
	act: T.Callable = nn.relu
	output_act: bool = False
	bias: bool = True
	bias_force: bool = False
	bn: bool = False
	dw_kernel_size: T.Optional[int] = None

	@nn.compact
	def __call__(self, input, training: bool = True):
		reshape = False
		if self.dw_kernel_size and input.ndim == 3:
			bs, n_tokens, in_dim = input.shape
			img_size = int(sqrt(n_tokens))
			input = jnp.reshape(input, (bs, img_size, img_size, in_dim))
			reshape = True

		hidden_dim, out_dim = get_dims(
			in_dim=input.shape[-1],
			hidden_dim_expansion_factor=self.hidden_dim_expansion_factor,
			hidden_dim=self.hidden_dim,
			out_dim=self.out_dim,
			)
		output = nn.Dense(
			features=hidden_dim,
			use_bias=self.bias_force or (self.bias and not self.bn),
			)(input)
		output = Conv(
			kernel_size=self.dw_kernel_size,
			groups='dw',
			bias=self.bias_force or (self.bias and not self.bn),
			)(output) if self.dw_kernel_size else output
		output = nn.BatchNorm(
			use_running_average=not training,
			)(output) if self.bn else output
		output = self.act(output)
		output = nn.Dense(
			features=out_dim,
			use_bias=self.bias,
			)(output)
		output = self.act(output) if self.output_act else output

		output = jnp.reshape(output, (bs, n_tokens, in_dim)) if reshape else output
		return output


class ConvMLP(nn.Module):
	"""
	Multilayer perceptron with one hidden layer, 
	implenented using convolutions.

	Args:
		out_dim (T.Optional[int]): Number of output channels. If None,
		it is set to the number of input channels.
		Default is None.
		hidden_dim_expansion_factor (T.Optional[float]): Factor of expansion for the 
		hidden layer. If None, hidden_dim must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
		hidden_dim (T.Optional[int]): Number of channels in the hidden layer.
		If None, hidden_dim_expansion_factor must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
		kernel_size (T.Union[T.Tuple[int, int], int]): Kernel size.
		If an int, this value is used along both spatial dimensions.
		Default is 3.
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
		act (T.Callable): Activation for the hidden layer.
		Default is nn.relu.
		output_act (bool): Whether to apply activation on the output.
		Default is False. 
		bias (bool): Whether the convolutions should have a bias term. If bn is True,
		the first convolution would not have a bias term.
		Default is True.
		bias_force (bool): Whether to force the first convolution to have a bias term 
		even if bn is True.
		Default is False.
		bn (bool): Whether to apply batch normalization before the activation.
		If False, the training argument is ignored.
		Default is False.
	"""
	out_dim: T.Optional[int] = None
	hidden_dim_expansion_factor: T.Optional[float] = None
	hidden_dim: T.Optional[int] = None
	kernel_size: T.Union[T.Tuple[int, int], int] = 1
	stride: T.Union[T.Tuple[int, int], int] = 1
	padding: T.Optional[T.Union[str, int]] = None
	groups: T.Union[int, str] = 1
	dilation: int = 1
	act: T.Callable = nn.relu
	output_act: bool = False
	bias: bool = True
	bias_force: bool = True
	bn: bool = False

	@nn.compact
	def __call__(self, input, training: bool = True):
		hidden_dim, out_dim = get_dims(
			in_dim=input.shape[-1],
			hidden_dim_expansion_factor=self.hidden_dim_expansion_factor,
			hidden_dim=self.hidden_dim,
			out_dim=self.out_dim,
			)
		output = Conv(
			out_dim=hidden_dim,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding,
			groups=self.groups,
			dilation=self.dilation,
			bias=self.bias_force or (self.bias and not self.bn),
			)(input)
		output = nn.BatchNorm(
			use_running_average=not training,
			)(output) if self.bn else output
		output = self.act(output)
		output = Conv(
			out_dim=out_dim,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding,
			groups=self.groups,
			dilation=self.dilation,
			bias=self.bias_force or self.bias,
			)(output)
		output = self.act(output) if self.output_act else output
		return output


class TransformerMLP(nn.Module):
	"""
	Transformer-style MLP with layer normalization, LayerScale,
	and a residual connection.

	Args:
		out_dim (T.Optional[int]): Number of output neurons. If None,
		it is set to the number of input channels.
		Default is None.
		hidden_dim_expansion_factor (T.Optional[float]): Factor of expansion for the 
		hidden layer. If None, hidden_dim must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
		hidden_dim (T.Optional[int]): Number of neurons in the hidden layer.
		If None, hidden_dim_expansion_factor must be provided. If both
		hidden_dim_expansion_factor and hidden_dim are provided, the latter takes precedence.
		Default is None.
		act (T.Callable): Activation for the hidden layer.
		Default is gelu.
		dw_kernel_size (T.Optional[int]): Kernel size of a depthwise convolution
		applied immediately after the first linear layer. If None,
		no depthwise convolution is applied.
		Default is None.
		layer_norm_eps (T.Optional[float]): Epsilon value passed to 
		layer normalization. If None, no normalization is applied.
		Default is 1e-6.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, no LayerScale
		is applied.
		Default is None.
		residual (bool): Whether a residual summation should be applied
		if the input and output shapes are identical.
		Default is True.
	"""
	out_dim: T.Optional[int] = None
	hidden_dim_expansion_factor: T.Optional[float] = 4
	hidden_dim: T.Optional[int] = None
	act: T.Callable = gelu
	dw_kernel_size: T.Optional[int] = None
	layer_norm_eps: T.Optional[float] = 1e-6
	layer_scale_init_value: T.Optional[float] = None
	residual: bool = True

	@nn.compact
	def __call__(self, input):
		output = nn.LayerNorm(
			epsilon=self.layer_norm_eps,
			)(input) if self.layer_norm_eps else input
		output = MLP(
			out_dim=self.out_dim,
			hidden_dim_expansion_factor=self.hidden_dim_expansion_factor,
			hidden_dim=self.hidden_dim,
			act=self.act,
			dw_kernel_size=self.dw_kernel_size,
			)(output)
		output = LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		return input+output if self.residual and input.shape == output.shape else output
