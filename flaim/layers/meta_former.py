"""
MetaFormer block by Yu et al.

References:
- Yu et al. MetaFormer Is Actually What You Need for Vision.
"""


__all__ = [
	'MetaFormerBlock',
	]


import typing as T

from flax import linen as nn

from .acts import gelu
from .layer_scale import LayerScale
from .mlp import TransformerMLP


class MetaFormerBlock(nn.Module):
	"""
	MetaFormer block.

	Args:
		token_mixer (T.Callable): T.Callable returning the token mixer.
		mlp_hidden_dim_expansion_factor (float): Factor of expansion for the 
		hidden layer of the MLP.
		Default is 4.
		act (T.Callable): Activation for the hidden layer of the MLP.
		Default is gelu.
		dw_kernel_size (T.Optional[int]): Kernel size of a depthwise convolution
		applied immediately after the first linear layer of the MLP. If None,
		no depthwise convolution is applied.
		Default is None.
		layer_norm_eps (T.Optional[float]): Epsilon value passed to 
		layer normalization. If None, no normalization is applied.
		Default is 1e-6.
		layer_scale_init_value (T.Optional[float]): Value
		for initializing LayerScale. If None, no LayerScale
		is applied.
		Default is None.
	"""
	token_mixer: T.Callable
	mlp_hidden_dim_expansion_factor: float = 4.
	act: T.Callable = gelu
	dw_kernel_size: T.Optional[int] = None
	layer_norm_eps: T.Optional[float] = 1e-6
	layer_scale_init_value: T.Optional[float] = None

	@nn.compact
	def __call__(self, input):
		output = nn.LayerNorm(
			epsilon=self.layer_norm_eps,
			)(input) if self.layer_norm_eps else input
		output = self.token_mixer()(output)
		output = LayerScale(
			init_value=self.layer_scale_init_value,
			)(output)
		output = input+output

		output = TransformerMLP(
			hidden_dim_expansion_factor=self.mlp_hidden_dim_expansion_factor,
			act=self.act,
			dw_kernel_size=self.dw_kernel_size,
			layer_norm_eps=self.layer_norm_eps,
			layer_scale_init_value=self.layer_scale_init_value,
			)(output)

		return output
