"""
Registry for tracking flaim models and accessing them.
"""


import re
import typing as T

import jax
from flax import linen as nn

from .model import Model


MODELS = {}
def register_models(
	get_configs: T.Callable,
	) -> T.Callable:
	"""
	Decorator for registering models by adding them to MODELS.

	Args:
		get_models (T.Callable): T.Callable that returns
		the model class and a dictionary of model configurations.

	Returns (T.Callable): get_models unmodified.
	"""
	model_cls, model_configs = get_configs()
	for model_name in model_configs:
		MODELS[model_name] = Model(
			model_cls=model_cls,
			**model_configs[model_name],
			)
	return get_configs


def get_model(
	model_name: str,
	pretrained: T.Union[str, int, bool] = True,
	input_size: int = 224,
	jit: bool = True,
	prng: T.Optional[jax.random.KeyArray] = None,
	n_classes: int = 0,
	) -> T.Union[T.Tuple[nn.Module, T.Dict], T.Tuple[nn.Module, T.Dict, T.Dict]]:
	"""
	Returns a flaim model and its parameters.

	Args:
		model_name (str): Name of model.
		pretrained (T.Union[str, int, bool]): If a string, the collection of pre-trained
		parameters of this name is returned. If an int, a collection of pre-trained
		parameters of this input size is returned. If True, a default collection
		of pre-trained parameters is returned. If False, the model's parameters are
		randomly initialized.
		Default is True.
		input_size (int): Input size the model expects. This argument is used only
		when pretrained is False.
		Default is 224.
		jit (bool): Whether to JIT the initialization function.
		Default is True.
		prng (T.Optional[jax.random.KeyArray]): PRNG key for initializing the model's parameters.
		If None, a key with a seed of 0 is created.
		Default is None.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.

	Returns (T.Union[T.Tuple[nn.Module, T.Dict], T.Tuple[nn.Module, T.Dict, T.Dict]]):
	The model, its parameters, and the normalization statistics associated with the
	parameters if pretrained is not False.
	"""
	if model_name not in MODELS:
		raise ValueError(f'{model_name} not recognized')

	model = MODELS[model_name]
	return model(
		pretrained=pretrained,
		input_size=input_size,
		jit=jit,
		prng=prng,
		n_classes=n_classes,
		)


def list_models(
	model_pattern: str = '',
	params_pattern: T.Union[str, int] = '',
	) -> T.List[T.Tuple[str, str]]:
	"""
	Lists available models and pre-trained parameters in pairs of
	(name of model, name of pre-trained parameters).

	Args:
		model_pattern (str): If not an empty string, only models containing
		this regex pattern are returned.
		Default is ''.
		params_pattern (T.Union[str, int]): If not an empty string, only pre-trained parameters
		containing this regex pattern are returned. If an int, only pre-trained parameters
		of this input size are returned.
		Default is ''.

	Returns (T.List[T.Tuple[str, str]]): List of tuples of form
	(name of model, name of pre-trained parameters) conforming to
	model_pattern and params_pattern.
	"""
	if isinstance(params_pattern, int):
		params_pattern = f'{params_pattern}$'

	models = []
	for model_name in list(MODELS):
		if re.search(model_pattern, model_name):
			for params_name in MODELS[model_name].params:
				if re.search(params_pattern, params_name):
					models.append((model_name, params_name))

	return models
