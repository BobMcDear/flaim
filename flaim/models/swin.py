"""
Shifted window attention transformer (Swin transformer).
"""


import typing as T
from functools import partial
from math import sqrt

from flax import linen as nn
from jax import numpy as jnp

from .. import layers
from .factory import register_configs


def get_mask(
	img_size: int,
	window_size: int,
	shift: int,
	) -> jnp.ndarray:
	"""
	Gets mask for shifted window atention.

	Args:
		img_size (int): Image size. This value is used
		along both spatial dimensions.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		Default is 7.
		shift (int): Shift size for cyclic shift.
	"""
	mask = jnp.zeros((1, img_size, img_size, 1))
	slices = (
		slice(0, -window_size),
		slice(-window_size, -shift),
		slice(-shift, None),
		)
	cnt = 0
	
	for slice_h in slices:
		for slice_w in slices:
			mask = mask.at[:, slice_h, slice_w, :].set(cnt)
			cnt += 1

	mask = layers.window_partition(mask, window_size)
	mask = jnp.reshape(mask, (-1, window_size ** 2))

	mask = jnp.expand_dims(mask, 1) - jnp.expand_dims(mask, 2)
	mask = jnp.where(mask != 0, float(-100.0), float(0.0))

	return mask


class Mask(nn.Module):
	"""
	Masking module for shifted window atention.

	Args:
		img_size (int): Image size. This value is used
		along both spatial dimensions.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		shift (int): Shift size for cyclic shift. This value
		is used along both spatial dimensions.
		If 0, no masking transpires.
	"""
	img_size: int
	window_size: int
	shift: int

	@nn.compact
	def __call__(self, input):
		if self.shift == 0:
			return input

		n_heads, n_tokens = input.shape[-3:-1]
		mask = self.variable(
			col='mask',
			name='mask',
			init_fn=lambda: get_mask(
				window_size=self.window_size,
				img_size=self.img_size,
				shift=self.shift,
				),
			).value
	
		output = jnp.reshape(input, (-1, len(mask), n_heads, n_tokens, n_tokens))
		mask = jnp.expand_dims(mask, (0, 2))
		output = jnp.reshape(mask+output, (-1, n_heads, n_tokens, n_tokens))

		return output


class WindowMHSA(nn.Module):
	"""
	Multi-headed self-attention, with support for
	shifted or non-shifted windows.

	Args:
		n_heads (int): Number of heads.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		shift (int): Shift size for cyclic shift. This value
		is used along both spatial dimensions. If 0, no shifting transpires.
		Default is None.
	"""
	n_heads: int
	window_size: int
	shift: int

	@nn.compact
	def __call__(self, input):
		n_tokens, token_dim = input.shape[-2:]
		window_size = self.window_size
		shift = self.shift
		img_size = int(sqrt(n_tokens))

		if img_size <= window_size:
			window_size = img_size
			shift = 0

		output = jnp.reshape(input, (-1, img_size, img_size, token_dim))
		output = layers.cyclic_shift(output, shift)
		output = layers.window_partition(output, window_size)
		output = jnp.reshape(output, (-1, window_size ** 2, token_dim))

		output = layers.MHSA(
			to_qkv=self.n_heads,
			pre_softmax=lambda: nn.Sequential([
				layers.RelPosEmbed(
					n_heads=self.n_heads,
					window_size=window_size,
					class_token=False,
					),
				Mask(
					img_size=img_size,
					window_size=window_size,
					shift=shift,
					),
				])
			)(output)

		output = layers.window_merge(
			output,
			window_size=window_size,
			img_size=img_size,
			)
		output = layers.cyclic_shift(output, -shift)
		output = jnp.reshape(output, input.shape)

		return output


class PatchMerge(nn.Module):
	"""
	Merges patches for downsampling.
	"""
	@nn.compact
	def __call__(self, input):
		n_tokens, token_dim = input.shape[-2:]
		img_size = int(sqrt(n_tokens))
		
		output = jnp.reshape(input, (-1, img_size, img_size, token_dim))
		s1 = output[:, 0::2, 0::2, :]
		s2 = output[:, 1::2, 0::2, :]
		s3 = output[:, 0::2, 1::2, :]
		s4 = output[:, 1::2, 1::2, :]

		output = jnp.concatenate([s1, s2, s3, s4], -1)
		output = jnp.reshape(output, (len(output), -1, 4*token_dim))

		output = nn.LayerNorm(
			epsilon=1e-5,
			)(output)
		output = nn.Dense(
			features=2*token_dim,
			use_bias=False,
			)(output)

		return output


class SwinStage(nn.Module):
	"""
	Swin stage.

	Args:
		depth (int): Depth.
		n_heads (int): Number of heads.
		window_size (int): Window size for relative position
		embedding and window attention. This value is used
		along both spatial dimensions.
		Default is 7.
		downsample (bool): Whether to downsample.
		Default is False.
	"""
	depth: int
	n_heads: int
	window_size: int = 7
	downsample: bool = False

	@nn.compact
	def __call__(self, input):
		for block_ind in range(self.depth):
			input = layers.MetaFormerBlock(
				token_mixer=partial(
					WindowMHSA,
					n_heads=self.n_heads,
					window_size=self.window_size,
					shift=self.window_size//2 if block_ind%2 else 0,
					),
				layer_norm_eps=1e-5,
				)(input)
		
		if self.downsample:
			input = PatchMerge()(input)

		return input


class Swin(nn.Module):
	"""
	Shifted window attention transformer.

	Args:
		depths (T.Tuple[int, ...]): Depth of each stage.
		token_dim (int): Token dimension.
		n_heads (T.Tuple[int, ...]): Number of heads of each
		stage.
		patch_size (int): Patch size. This value is used along
		both spatial dimensions.
		Default is 4.
		window_size (T.Union[T.Tuple[int, ...], int]): Window size for 
		relative position embedding and window attention, used along
		both spatial dimensions. If an int, this value is used for every stage,
		and if a T.List, it must contain the window size for each stage.
		Default is 7.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
	"""
	depths: T.Tuple[int, ...]
	token_dim: int
	n_heads: T.Tuple[int, ...]
	patch_size: int = 4
	window_size: T.Union[T.Tuple[int, ...], int] = 7
	n_classes: int = 0

	@nn.compact
	def __call__(self, input):
		output = layers.PatchEmbed(
			token_dim=self.token_dim,
			patch_size=self.patch_size,
			layer_norm_eps=1e-5,
			)(input)
		self.sow(
			col='intermediates',
			name='stage_0',
			value=output,
			)
		
		window_size = layers.tuplify(self.window_size, len(self.depths))
		for stage_ind in range(len(self.depths)):
			output = SwinStage(
				depth=self.depths[stage_ind],
				n_heads=self.n_heads[stage_ind],
				window_size=window_size[stage_ind],
				downsample=True if stage_ind < len(self.depths)-1 else False,
				)(output)
			self.sow(
				col='intermediates',
				name=f'stage_{stage_ind+1}',
				value=output,
				)
		
		output = layers.Head(
			n_classes=self.n_classes,
			layer_norm_eps=1e-5,
			norm_first=True,
			)(output)

		return output


@register_configs
def get_swin_configs() -> T.Tuple[T.Type[Swin], T.Dict]:
	"""
	Gets configurations for all available
	Swin models.

	Returns (T.Tuple[T.Type[Swin], T.Dict]): The Swin class and
	configurations of all available models.
	"""
	configs = {
		'swin_tiny_window7_224': {
			'depths': (2, 2, 6, 2),
			'token_dim': 96,
			'n_heads': (3, 6, 12, 24),
			},
		'swin_small_window7_224': {
			'depths': (2, 2, 18, 2),
			'token_dim': 96,
			'n_heads': (3, 6, 12, 24),
			},
		'swin_base_window7_224': {
			'depths': (2, 2, 18, 2),
			'token_dim': 128,
			'n_heads': (4, 8, 16, 32),
			},
		'swin_large_window7_224': {
			'depths': (2, 2, 18, 2),
			'token_dim': 192,
			'n_heads': (6, 12, 24, 48),
			},
		'swin_base_window12_384': {
			'depths': (2, 2, 18, 2),
			'token_dim': 128,
			'n_heads': (4, 8, 16, 32),
			'window_size': 12,
			},
		'swin_large_window12_384': {
			'depths': (2, 2, 18, 2),
			'token_dim': 192,
			'n_heads': (6, 12, 24, 48),
			'window_size': 12,
			},
		'swin_base_window7_224_in22k': {
			'depths': (2, 2, 18, 2),
			'token_dim': 128,
			'n_heads': (4, 8, 16, 32),
			'window_size': 7,
			},
		'swin_large_window7_224_in22k': {
			'depths': (2, 2, 18, 2),
			'token_dim': 192,
			'n_heads': (6, 12, 24, 48),
			'window_size': 7,
			},
		'swin_base_window12_384_in22k': {
			'depths': (2, 2, 18, 2),
			'token_dim': 128,
			'n_heads': (4, 8, 16, 32),
			'window_size': 12,
			},
		'swin_large_window12_384_in22k': {
			'depths': (2, 2, 18, 2),
			'token_dim': 192,
			'n_heads': (6, 12, 24, 48),
			'window_size': 12,
			},
		'swin_s3_tiny_224': {
			'depths': (2, 2, 6, 2),
			'token_dim': 96,
			'n_heads': (3, 6, 12, 24),
			'window_size': (7, 7, 14, 7),
			},
		'swin_s3_small_224': {
			'depths': (2, 2, 18, 2),
			'token_dim': 96,
			'n_heads': (3, 6, 12, 24),
			'window_size': (14, 14, 14, 7),
			},
		'swin_s3_base_224': {
			'depths': (2, 2, 30, 2),
			'token_dim': 96,
			'n_heads': (3, 6, 12, 24),
			'window_size': (7, 7, 14, 7),
			},
		}
	return Swin, configs
