"""
Large kernel attention (LKA) by Guo et al.

References:
- Guo et al. Visual Attention Network
"""


__all__ = [
	'LKA',
	]


from flax import linen as nn

from .conv import Conv


class LKA(nn.Module):
	"""
	Large kernel attention.

	Args:
		dw_kernel_size (int): Kernel size of
		the depthwise convolution.
		Default is 5.
		dilated_dw_kernel_size (int): Kernel size of
		the dilated depthwise convolution.
		Default is 7.
	"""
	dw_kernel_size: int = 5
	dilated_dw_kernel_size: int = 7

	@nn.compact
	def __call__(self, input):
		attention = Conv(
			kernel_size=self.dw_kernel_size,
			groups='dw',
			)(input)
		attention = Conv(
			kernel_size=self.dilated_dw_kernel_size,
			groups='dw',
			dilation=self.dilated_dw_kernel_size//2,
			)(attention)
		attention = Conv(
			kernel_size=1,
			)(attention)
		return input*attention
