"""
Tests flaim models against their timm counterparts to ensure consistency.
"""


import urllib.request
import typing as T
from io import BytesIO

import flax
import timm
import torch
import numpy as np
from PIL import Image
from jax import numpy as jnp

import flaim


def get_input_img(
	url: str = 'https://3.bp.blogspot.com/-W__wiaHUjwI/Vt3Grd8df0I/AAAAAAAAA78/7xqUNj8ujtY/s1600/image02.png',
	size: int = 224,
	) -> np.ndarray:
	"""
	Gets an image as input for vision models.

	Args:
		url (str): URL to download the image from.
		Default is 'https://3.bp.blogspot.com/-W__wiaHUjwI/Vt3Grd8df0I/AAAAAAAAA78/7xqUNj8ujtY/s1600/image02.png'.
		size (int): Size to which the image is resized.
		Default is 224.

	Returns (np.ndarray): The image as a NumPy array.
	"""
	data = BytesIO(urllib.request.urlopen(url).read())
	img = Image.open(data).resize((size, size))

	input = np.array(img)/255.
	input = np.expand_dims(input, axis=0)
	input = np.repeat(input, repeats=2, axis=0)

	return input


def normalize(
	input: np.ndarray,
	mean: T.Tuple[float, ...],
	std: T.Tuple[float, ...],
	) -> np.ndarray:
	"""
	Normalizes a NumPy array.

	Args:
		input (np.ndarray): NumPy array to normalize.
		mean (T.Tupe[float, ...]): Means for normalization.
		std (T.Tuple[float, ...]): Standard deviations for normalization.

	Returns (np.ndarray): Normalized input.
	"""
	return (input - np.array(mean)) / np.array(std)


@torch.no_grad()
def get_outputs(
	flaim_input: T.Any,
	flaim_model: flax.linen.Module,
	flaim_vars: T.Dict,
	timm_input: torch.Tensor,
	timm_model: torch.nn.Module,
	) -> T.Tuple[T.Any, torch.Tensor]:
	"""
	Gets the outputs of corresponding flaim and timm models.

	Args:
		flaim_input (T.Any): Input for the flaim model.
		flaim_model (flax.linen.Module): The flaim model.
		flaim_vars (T.Dict): The flaim model's parameters.
		timm_input (torch.Tensor): Corresponding input for the timm model.
		timm_model (torch.nn.Module): The corresponding timm model.

	Returns (T.Tuple[T.Any, torch.Tensor]): The outputs of flaim_model
	and timm_model.
	"""
	if 'batch_stats' in flaim_vars:
		flaim_output_inf = flaim_model.apply(flaim_vars, flaim_input, training=False, mutable=False)
		flaim_output_train, new_batch_stats = flaim_model.apply(flaim_vars, flaim_input, training=True, mutable=['batch_stats'])
		flaim_output = jnp.concatenate([flaim_output_inf, flaim_output_train])

		timm_output_inf = timm_model.eval()(timm_input)
		timm_output_train = timm_model.train()(timm_input)
		timm_output = torch.cat([timm_output_inf, timm_output_train])

	else:
		flaim_output = flaim_model.apply(flaim_vars, flaim_input)
		timm_output = timm_model.eval()(timm_input)

	return flaim_output, timm_output


@torch.no_grad()
def test_outputs(
	input: np.ndarray,
	flaim_model_name: str,
	flaim_params_name: str,
	timm_model_name: str,
	) -> None:
	"""
	Compares the outputs of corresponding flaim and timm models.

	Args:
		input (np.ndarray): NumPy array to be used as input data.
		flaim_model_name (str): Name of the flaim model.
		flaim_params_name (str): Name of the flaim model's pre-trained parameters.
		timm_model_name (str): Name of the corresponding timm model.
	"""
	flaim_model, flaim_vars, flaim_norm_stats = flaim.get_model(
		model_name=flaim_model_name,
		pretrained=flaim_params_name,
		jit=False,
		n_classes=-1,
		)
	timm_model = timm.create_model(
		model_name=timm_model_name,
		pretrained=True,
		num_classes=0,
		)
	timm_norm_stats = {'mean': timm_model.default_cfg['mean'], 'std': timm_model.default_cfg['std']}

	flaim_input = jnp.array(normalize(input, **flaim_norm_stats), dtype=jnp.float32)
	timm_input = torch.tensor(normalize(input, **timm_norm_stats), dtype=torch.float32).permute(0, 3, 1, 2)

	flaim_output, timm_output = get_outputs(
		flaim_input=flaim_input,
		flaim_model=flaim_model,
		flaim_vars=flaim_vars,
		timm_input=timm_input,
		timm_model=timm_model,
		)
	assert jnp.allclose(flaim_output, jnp.array(timm_output), atol=1e-4), 'Outputs not equal'


def main() -> None:
	"""
	Tests flaim models against their timm counterparts to ensure consistency.
	"""
	flaim_to_timm = {
		('cait_xxsmall24', 'in1k_224'): 'cait_xxs24_224',
		('convmixer20_1024d_patch14_kernel9', 'in1k_224'): 'convmixer_1024_20_ks9_p14',
		('convnext_atto', 'in1k_224'): 'convnext_atto.d2_in1k',
		('convnextv2_atto', 'fcmae_in1k_224'): 'convnextv2_atto.fcmae',
		('davit_tiny', 'in1k_224'): 'davit_tiny.msft_in1k',
		('efficientnetv2_small', 'in1k_300'): 'tf_efficientnetv2_s.in1k',
		('gcvit_xxtiny', 'in1k_224'): 'gcvit_xxtiny',
		('maxvit_tiny', 'in1k_224'): 'maxvit_tiny_tf_224.in1k',
		('nest_tiny', 'in1k_224'): 'jx_nest_tiny',
		('pit_tiny', 'in1k_224'): 'pit_ti_224',
		('pvtv2_b0', 'in1k_224'): 'pvt_v2_b0',
		('regnetx_200mf', 'in1k_224'): 'regnetx_002',
		('regnety_200mf', 'in1k_224'): 'regnety_002',
		('resnet18', 'in1k_224'): 'resnet18',
		('resnet26', 'in1k_224'): 'resnet26',
		('resnet18d', 'in1k_224'): 'resnet18d',
		('resnet50d', 'in1k_224'): 'resnet50d',
		('resnet10t', 'in1k_176'): 'resnet10t',
		('resnet14t', 'in1k_176'): 'resnet14t',
		('resnext50_32x4d', 'in1k_224'): 'resnext50_32x4d',
		('seresnext50_32x4d', 'in1k_224'): 'seresnext50_32x4d',
		('ecaresnet26t', 'in1k_256'): 'ecaresnet26t',
		('resnetrs50', 'in1k_160'): 'resnetrs50',
		('resnest14_2s1x64d', 'in1k_224'): 'resnest14d',
		('swin_tiny_window7', 'in1k_224'): 'swin_tiny_patch4_window7_224',
		('vit_tiny_patch16', 'augreg_in22k_224'): 'vit_tiny_patch16_224.augreg_in21k',
		('vit_base_clip_patch32', 'clip_laion2b_224'): 'vit_base_patch32_clip_224.laion2b',
		('deit3_small_patch16', 'in1k_224'): 'deit3_small_patch16_224',
		('beit_base_patch16', 'beit_in22k_ft_in22k_224'): 'beit_base_patch16_224.in22k_ft_in22k',
		}

	input = get_input_img()
	for flaim_model_name, flaim_params_name in flaim_to_timm:
		print(flaim_model_name, flaim_params_name)
		test_outputs(
			input=input,
			flaim_model_name=flaim_model_name,
			flaim_params_name=flaim_params_name,
			timm_model_name=flaim_to_timm[(flaim_model_name, flaim_params_name)],
			)


if __name__ == '__main__':
	main()
