"""
Recursive gated convolution (g^n convolution) by Rao et al.

References:
- Rao et al. HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions.
"""


__all__ = [
	'GnConv',
	]


from flax import linen as nn
from jax import numpy as jnp

from .lin_norm_act import Conv


class GnConv(nn.Module):
	"""
	Recursive gated convolution.

	Args:
		out_dim (int): Number of output channels.
		order (int): Order of g^n convolution.
		Default is 5.
		scale (float): Scale factor for the output
		of the depthwise convolution.
		Default is 1/3.
	"""
	out_dim: int
	order: int = 5
	scale: float = 1/3

	@nn.compact
	def __call__(self, input):
		out_dims = [self.out_dim // (2 ** (self.order-1))]
		split_dims = [out_dims[0]]
		for ord_ind in range(self.order-2, -1, -1):
			out_dims.append(self.out_dim // (2 ** ord_ind))
			split_dims.append(split_dims[-1] + out_dims[-1])

		output = Conv(
			out_dim=2*self.out_dim,
			kernel_size=1,
			)(input)
		p0, q0 = jnp.split(
			ary=output,
			indices_or_sections=(split_dims[0],),
			axis=-1,
			)

		ps = self.scale * Conv(
			kernel_size=7,
			groups='dw',
			)(q0)
		ps = jnp.split(
			ary=ps,
			indices_or_sections=split_dims[:-1],
			axis=-1,
			)
		
		output = p0 * ps[0]
		for ord_ind in range(self.order-1):
			output = ps[ord_ind+1] * Conv(
				out_dim=out_dims[ord_ind+1],
				kernel_size=1,
				)(output)
		output = Conv(
			kernel_size=1,
			)(output)
			
		return output
