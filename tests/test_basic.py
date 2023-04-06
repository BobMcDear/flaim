"""
Tests models to verify a forward pass can be performed,
ensure the soundness of pre-trained parameters, and check
JIT compatibility.
"""


import argparse
import typing as T

from flax import linen as nn
from jax import numpy as jnp

import flaim


def str_to_bool(
	input: str,
	) -> bool:
	"""
	Converts a string to a boolean by returning
	True if it is 'True' and False if it is 'False'.

	Args:
		input (str): Input to be converted to a boolean.

	Returns (bool): The input converted into a boolean.
	"""
	if input == 'True':
		return True

	elif input == 'False':
		return False

	else:
		raise ValueError(f'{input} not valid for conversion into a boolean')


def parse_args() -> argparse.Namespace:
	"""
	Parses arguments.
	"""
	parser = argparse.ArgumentParser(
		description='Tests models to ensure compatibility with JIT, \
		check the soundness of pre-trained parameters, and verify a forward pass can be performed.',
		)

	parser.add_argument(
		'--small_sample',
		type=str_to_bool,
		default='True',
		help='Whether to test a small sample of the models.',
		)
	parser.add_argument(
		'--pretrained',
		type=str_to_bool,
		default='True',
		help='Whether to test the pre-trained parameters.',
		)
	parser.add_argument(
		'--jit',
		type=str_to_bool,
		default='False',
		help='Whether to JIT.',
		)

	return parser.parse_args()


def test_forward_pass(
	input: T.Any,
	model: nn.Module,
	vars: T.Dict,
	) -> None:
	"""
	Tests a model's forward pass.

	Args:
		input (T.Any): Input passed to the model.
		model (nn.Module): Model to test.
		vars (T.Dict): The model's parameters.
	"""
	if 'batch_stats' in vars:
		model.apply(vars, input, training=True, mutable=['batch_stats'])
		model.apply(vars, input, training=False, mutable=False)

	else:
		model.apply(vars, input)


def main(
	small_sample: bool = True,
	pretrained: bool = True,
	jit: bool = True,
	) -> None:
	"""
	Tests models to verify a forward pass can be performed,
	ensure the soundness of pre-trained parameters, and check
	JIT compatibility.

	Args:
		small_sample (bool): Whether to test a small sample
		of the models.
		Default is True.
		pretrained (bool): Whether to test the pre-trained parameters.
		Default is True.
		jit (bool): Whether to JIT.
		Default is True.
	"""
	if small_sample:
		models = [
			('cait_xxsmall24', 'in1k_224'),
			('convmixer20_1024d_patch14_kernel9', 'in1k_224'),
			('convnext_atto', 'in1k_224'),
			('convnextv2_atto', 'fcmae_in1k_224'),
			('davit_tiny', 'in1k_224'),
			('efficientnetv2_small', 'in1k_300'),
			('gcvit_xxtiny', 'in1k_224'),
			('maxvit_tiny', 'in1k_224'),
			('nest_tiny', 'in1k_224'),
			('pit_tiny', 'in1k_224'),
			('pvtv2_b0', 'in1k_224'),
			('regnetx_200mf', 'in1k_224'),
			('regnety_200mf', 'in1k_224'),
			('resnet18', 'in1k_224'),
			('resnet26', 'in1k_224'),
			('resnet18d', 'in1k_224'),
			('resnet50d', 'in1k_224'),
			('resnet10t', 'in1k_176'),
			('resnet14t', 'in1k_176'),
			('resnext50_32x4d', 'in1k_224'),
			('seresnext50_32x4d', 'in1k_224'),
			('ecaresnet26t', 'in1k_256'),
			('resnetrs50', 'in1k_160'),
			('resnest14_2s1x64d', 'in1k_224'),
			('swin_tiny_window7', 'in1k_224'),
			('vit_tiny_patch16', 'augreg_in22k_224'),
			('vit_base_clip_patch32', 'clip_laion2b_224'),
			('deit3_small_patch16', 'in1k_224'),
			('beit_base_patch16', 'beit_in22k_ft_in22k_224'),
			]

	else:
		models = flaim.list_models()

	for model_name, params in models:
		print(model_name, params)
		model, vars = flaim.get_model(
			model_name=model_name,
			pretrained=pretrained and params,
			n_classes=10,
			jit=jit,
			)[:2]

		if pretrained:
			input_size = int(params[-3:])
			input = jnp.ones((1, input_size, input_size, 3))
			test_forward_pass(input, model, vars)


if __name__ == '__main__':
	args = parse_args()
	main(
		small_sample=args.small_sample,
		pretrained=args.pretrained,
		jit=args.jit,
		)
