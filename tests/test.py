"""
Tests models to ensure compatibility with JIT,
check the soundness of pre-trained parameters, and
verify a forward pass can be performed.
"""

import argparse

import flaim


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
		type=bool,
		default=True,
		help='Whether to test a small sample of the models, with one representative from each family of models.',
		)
	parser.add_argument(
		'--pretrained',
		type=bool,
		default=True,
		help='Whether to test with pre-trained parameters.',
		)
	parser.add_argument(
		'--jit',
		type=bool,
		default=True,
		help='Whether to JIT.',
		)
	
	return parser.parse_args()
	


def main(
	small_sample: bool = True,
	pretrained: bool = True,
	jit: bool = True,
	) -> None:
	"""
	Tests models to ensure compatibility with JIT,
	check the soundness of pre-trained parameters, and
	verify a forward pass can be performed.

	Args:
		small_sample (bool): Whether to test a small sample
		of the models, with one representative from each
		family of models.
		Default is True.
		pretrained (bool): Whether to test with pre-trained parameters.
		Default is True.
		jit (bool): Whether to JIT.
		Default is True.
	"""
	if small_sample:
		model_names = [
			'cait_xxsmall24_224',
			'convmixer20_1024d_patch14_kernel9',
			'convnext_xxxnano',
			'convnextv2_atto_fcmae',
			'efficientnetv2_small',
			'gcvit_xxtiny_224',
			'hornet_tiny',
			'maxvit_tiny_224',
			'nest_tiny_224',
			'pvtv2_b0',
			'resnet18',
			'resnet26',
			'resnet18d',
			'resnet50d',
			'resnet10t',
			'resnet14t',
			'resnext50_32x4d',
			'seresnext50_32x4d',
			'ecaresnet26t',
			'resnetrs50',
			'skresnet18',
			'skresnext50_32x4d',
			'resnest14_2s1x64d',
			'swin_tiny_window7_224',
			'van_b0',
			'vgg11',
			'vgg11_bn',
			'vit_tiny_patch16_224',
			'vit_base_clip_patch32_224_laion2b',
			'deit3_small_patch16_224',
			'beit_base_patch16_224',
			'xcit_nano12_patch16_224',
			]
	
	else:
		model_names = flaim.list_models()


	for model_name in model_names:
		print(model_name)
		flaim.get_model(
			model_name=model_name,
			pretrained=pretrained,
			n_classes=10,
			jit=jit,
			)


if __name__ == '__main__':
	args = parse_args()
	main(
		small_sample=args.small_sample,
		pretrained=args.pretrained,
		jit=args.pretrained,
		)
