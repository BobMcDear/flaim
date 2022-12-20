"""
Some of the smaller modules used by the vision transformer by Dosovitskiy et al.:
- PatchEmbed: Patch embedding.
- ClassToken: ClassToken module.
- AbsPosEmbed: Absolution position embedding.

References: 
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
"""


__all__ = [
	'PatchEmbed',
	'ClassToken',
	'AbsPosEmbed',
	]


import typing as T

import jax
from flax import linen as nn
from jax import numpy as jnp

from .conv import Conv
from .identity import identity


class PatchEmbed(nn.Module):
	"""
	Patch embedding.

	Args:
		token_dim (T.Optional[int]): Token dimension. If None, it is
		set to the dimensionality of the input.
		Default is None.
		patch_size (T.Union[T.Tuple[int, int], int]): Patch size. If an
		int, this value is used along both spatial dimensions.
		Default is 16.
		patch_stride (T.Optional[T.Union[T.Tuple[int, int], int]]): Stride of each patch. 
		If None, it is the same as the patch size. If an int,
		this value is used along both spatial dimensions.
		Default is None.
		patch_padding (T.Optional[T.Union[str, int]]): Padding for patches. 
		If an int, this value is used along both spatial dimensions. If None
		and patch_size and patch_stride are the same, it is set to 0. If None
		and patch_size and patch_stride are different, it is set so the spatial
		dimensions are exactly divided by patch_stride.
		Default is None.
		bias (bool): Whether the linear transformation should have a bias term.
		Default is True.
		layer_norm_eps (T.Optional[float]): Epsilon value passed to 
		layer normalization. If None, no normalization is applied, and norm_first
		is ignored.
		Default is None.
		norm_first (bool): Whether to apply layer normalization before the 
		linear transformation instead of after.
		Default is False.
		flatten (bool): Whether the output should be flattened along the spatial
		dimensions.
		Default is True.
	"""
	token_dim: T.Optional[int] = None
	patch_size: T.Union[T.Tuple[int, int], int] = 16
	patch_stride: T.Optional[T.Union[T.Tuple[int, int], int]] = None
	patch_padding: T.Optional[T.Union[str, int]] = None
	bias: bool = True
	layer_norm_eps: T.Optional[float] = None
	norm_first: bool = False
	flatten: bool = True

	@nn.compact
	def __call__(self, input):
		token_dim = self.token_dim or input.shape[-1]
		layer_norm = nn.LayerNorm(self.layer_norm_eps) if self.layer_norm_eps else identity
		patch_stride = self.patch_stride or self.patch_size
		patch_padding = 0 if self.patch_size == patch_stride else None
		
		if self.norm_first:
			output = layer_norm(input)
			output = Conv(
				out_dim=token_dim,
				kernel_size=self.patch_size,
				stride=patch_stride,
				padding=patch_padding,
				bias=self.bias,
				)(output)
		
		else:
			output = Conv(
				out_dim=token_dim,
				kernel_size=self.patch_size,
				stride=patch_stride,
				padding=patch_padding,
				bias=self.bias,
				)(input)
			output = layer_norm(output)
		
		if self.flatten:
			output = jnp.reshape(output, (len(input), -1, token_dim))

		return output


class ClassToken(nn.Module):
	"""
	Class token module.

	Args:
		concat (bool): Whether to concatenate the class
		token to the beginning of the input when returning the output.
		If False, only the class token is returned.
		Default is True.
	"""
	concat: bool = True

	@nn.compact
	def __call__(self, input):
		token_dim = input.shape[-1]
		class_token = self.param(
			name='class_token',
			init_fn=lambda prng: jnp.zeros((1, 1, token_dim)),
			)
		class_token = jnp.broadcast_to(class_token, (len(input), 1, token_dim))
		return jnp.concatenate((class_token, input), axis=1) if self.concat else class_token


class AbsPosEmbed(nn.Module):
	"""
	Learnable absolute position embedding.

	Args:
		n_axes (int): Absolute position embedding
		is applied along the last n_axes axes.
		Default is -2.
		add (bool): Whether to add position embedding to the input
		when returning the output. If False, only the position embedding
		is returned.
		Default is True.
	"""
	n_axes: int = -2
	add: bool = True

	@nn.compact
	def __call__(self, input):
		pos_embed = self.param(
			name='pos_embed',
			init_fn=lambda prng: jax.random.normal(prng, input.shape[self.n_axes:])
			)
		return input+pos_embed if self.add else pos_embed
