"""
VGG.
"""


import typing as T

from flax import linen as nn

from .. import layers
from .factory import register_configs


class VGG(nn.Module):
	"""
	VGG network.

	Args:
		out_dims (T.Tuple[T.Union[int, str], ...]): Number of output channels of
		each convolution-ReLU block. If 'm', max pooling is used.
		bn (bool): Whether to have batch normalization after every convolution.
		If False, the training argument is ignored.
		Default is False.
		n_classes (int): Number of output classes. If 0, there is no head,
		head_conv_mlp_dim is ignored, and the raw final features are returned.
		If -1, all stages of the  head, other than the final linear layer, are applied
		and the output returned.
	"""
	out_dims: T.Tuple[T.Union[int, str], ...]
	bn: bool = False
	n_classes: int = 0

	@nn.compact
	def __call__(self, input, training: bool = True):
		stage_ind = 0
		for width in self.out_dims:
			if isinstance(width, int):
				input = layers.ConvBNAct(
					out_dim=width,
					bias_force=True,
					bn=self.bn,
					act=nn.relu,
					)(input, training=training)
			
			elif width == 'm':
				self.sow(
					col='intermediates',
					name=f'stage_{stage_ind}',
					value=input,
					)
				input = layers.max_pool(
					input=input,
					kernel_size=2,
					stride=2,
					padding=0,
					)
				stage_ind += 1
		
		output = layers.Head(
			n_classes=self.n_classes,
			)(input)
			
		return output


@register_configs
def get_vgg_configs() -> T.Tuple[T.Type[VGG], T.Dict]:
	"""
	Gets configurations for all available
	VGG models.

	Returns (T.Tuple[T.Type[VGG], T.Dict]): VGG class and
	configurations of all models.
	"""
	configs = {
		'vgg11': {
			'out_dims': (64, 'm', 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'),
			},
		'vgg13': {
			'out_dims': (64, 64, 'm', 128, 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'),
			},
		'vgg16': {
			'out_dims': (64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', 512, 512, 512, 'm'),
			},
		'vgg19': {
			'out_dims': (64, 64, 'm', 128, 128, 'm', 256, 256, 256, 256, 'm', 512, 512, 512, 512, 'm', 512, 512, 512, 512, 'm'),
			},
		'vgg11_bn': {
			'out_dims': (64, 'm', 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'),
			'bn': True,
			},
		'vgg13_bn': {
			'out_dims': (64, 64, 'm', 128, 128, 'm', 256, 256, 'm', 512, 512, 'm', 512, 512, 'm'),
			'bn': True,
			},
		'vgg16_bn': {
			'out_dims': (64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', 512, 512, 512, 'm'),
			'bn': True,
			},
		'vgg19_bn': {
			'out_dims': (64, 64, 'm', 128, 128, 'm', 256, 256, 256, 256, 'm', 512, 512, 512, 512, 'm', 512, 512, 512, 512, 'm'),
			'bn': True,
			},
		}
	return VGG, configs
