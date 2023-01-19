"""
Squeeze-and-excitation (SE) by Hu et al.

References:
- Hu et al. Squeeze-and-Excitation Networks.
"""


__all__ = [
	'SE',
	]


import typing as T

from flax import linen as nn

from .mlp import MLP
from .pool import global_avg_pool


class SE(nn.Module):
	"""
	Squeeze-and-excitation.

	Args:
		reduction_factor (T.Optional[int]): Factor of reduction for the 
		hidden layer of the excitation module. If None, reduction_dim
		must be provided. If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is 16.
		reduction_dim (T.Optional[int]): Number of channels in the 
		hidden layer of the excitation module. If None, reduction_factor
		must be provided. If both reduction_factor and reduction_dim are provided,
		the latter takes precedence.
		Default is None.
		pool (T.Callable): Pooling method used for aggregating
		the activations of each channel.
		Default is global_avg_pool.
		act (T.Callable): Activation for the excitation module.
		Default is nn.relu.
		gate (T.Callable): Gating function used to normalize the attention
		scores.
		Default is nn.sigmoid.
		bias (bool): Whether the linear layers should have a bias term.
		Default is True.
		bn (bool): Whether to add batch normalization to the excitation module.
		If False, the training argument is ignored.
		Default is False.
		bottleneck (bool): If True, SE's usual MLP is used, i.e., one with 
		a single hidden layer. Otherwise, a single fully-connected layer 
		with no non-linearities is used, and reduction_factor, reduction_dim,
		act, and bn are ignored.
		Default is True.
		return_attn (bool): Whether to return the raw attention values instead 
		of gating them and multiplying them by the input. If True, gate is ignored.
		Default is False.
	"""
	reduction_factor: T.Optional[int] = 16
	reduction_dim: T.Optional[int] = None
	pool: T.Callable = global_avg_pool
	act: T.Callable = nn.relu
	gate: T.Callable = nn.sigmoid
	bias: bool = True
	bn: bool = False
	bottleneck: bool = True
	return_attn: bool = False

	@nn.compact
	def __call__(self, input, training: bool = True):
		attention = self.pool(input)

		if self.bottleneck:
			attention = MLP(
				hidden_dim_expansion_factor=1/self.reduction_factor if self.reduction_factor else None,
				hidden_dim=self.reduction_dim,
				act=self.act,
				bias=self.bias,
				bn=self.bn,
				)(attention, training=training)
		
		else:
			attention = nn.Dense(
				features=input.shape[-1],
				use_bias=self.bias,
				)(attention)
			
		if self.return_attn:
			return attention
		
		attention = self.gate(attention)
		return attention*input
