"""
Utilities for initializing flaim models and loading pre-trained parameters.
"""


import typing as T

import jax
from flax import traverse_util
from flax import linen as nn
from jax import numpy as jnp

from .hub import load_params


def init_model(
	model: nn.Module,
	input_size: int = 224,
	jit: bool = True,
	prng: T.Optional[jax.random.KeyArray] = None,
	):
	"""
	Initializes a model's parameters.

	Args:
		model (nn.Module): Model to initialize.
		input_size (int): Input size the model expects.
		Default is 224.
		jit (bool): Whether to JIT the model's initialization function.
		Default is True.
		prng (T.Optional[jax.random.KeyArray]): PRNG key for
		initializing the model. If None, a key with a seed of 0 is created.
		Default is None.
	"""
	init_fn = jax.jit(model.init) if jit else model.init
	return init_fn(prng or jax.random.PRNGKey(0), jnp.ones((1, input_size, input_size, 3)))


def merge_vars(
	vars: T.Dict,
	pretrained_vars: T.Dict,
	) -> T.Dict:
	"""
	Merges dictionaries of randomly-initialized and pre-trained variables,
	with the pre-trained variables taking precedence in case of duplicates.

	Args:
		vars (T.Dict): Randomly-initialized variables.
		pretrained_vars (T.Dict): Pre-trained variables.

	Returns (T.Dict): vars and pretrained_vars merged together.
	"""
	vars = traverse_util.flatten_dict(vars)
	pretrained_vars = traverse_util.flatten_dict(pretrained_vars)

	merged_vars = {
		**vars,
		**pretrained_vars,
		}
	return traverse_util.unflatten_dict(merged_vars)


class Model:
	"""
	Generic class for initializing flaim models and loading pre-trained parameters.

	Args:
		model_cls (type): Model type used to construct the model.
		model_config (T.Dict): Dictionary of arguments passed to model_cls
		for constructing the model.
		params (T.Optional[T.Dict]): Dictionary of configurations of available pre-trained parameters.
	"""
	def __init__(
		self,
		model_cls: type,
		model_args: T.Dict,
		params: T.Dict,
		) -> None:
		self.model_cls = model_cls
		self.model_args = model_args
		self.params = params

	def get_params_name(
		self,
		pretrained: T.Union[str, int, bool] = True,
		) -> str:
		"""
		Gets the name of the desired set of pre-trained parameters
		according to pretrained.

		Args:
			pretrained (T.Union[str, int, bool]): If a string, pretrained is returned as is.
			If an int, the name of a collection of pre-trained parameters of this input size is returned.
			If True, the name of a default collection of pre-trained parameters is returned.
			Default is True.

		Returns (str): The name of the desired set of pre-trained parameters.
		"""
		if pretrained is True:
			pretrained = list(self.params)[-1]

		elif isinstance(pretrained, int):
			size = pretrained
			pretrained = next((params for params in reversed(self.params) if int(params[-3:]) == size), None)
			if pretrained is None:
				raise ValueError(f'Pre-trained parameters of resolution {size} not available')

		elif pretrained not in self.params:
			raise ValueError(f'Pre-trained parameters {pretrained} not valid')

		return pretrained

	def get_params(
		self,
		pretrained: T.Union[str, int, bool] = True,
		n_classes: int = 0,
		) -> T.Tuple[T.Dict, T.Dict]:
		"""
		Gets the desired set of pre-trained parameters according to pretrained.

		Args:
			pretrained (T.Union[str, int, bool]): If a string, the collection of pre-trained
			parameters of this name is returned. If an int, a collection of pre-trained
			parameters of this input size is returned. If True, a default collection
			of pre-trained parameters is returned.
			Default is True.
			n_classes (int): Number of output classes. If 0, the head's parameters
			are removed.
			Default is 0.

		Returns (T.Tuple[T.Dict, T.Dict]): The desired set of pre-trained parameters
		and its associated normalization statistics.
		"""
		params_name = self.get_params_name(pretrained)
		params_config = self.params[params_name]
		pretrained_vars = load_params(
			repo_name=params_config['repo_name'],
			n_classes=n_classes,
			)
		return pretrained_vars, params_config['norm_stats']

	def __call__(
		self,
		pretrained: T.Union[str, int, bool] = True,
		input_size: int = 224,
		jit: bool = True,
		prng: T.Optional[jax.random.KeyArray] = None,
		n_classes: int = 0,
		**model_kwargs,
		) -> T.Union[T.Tuple[nn.Module, T.Dict], T.Tuple[nn.Module, T.Dict, T.Dict]]:
		"""
		Constructs and initializes the model.

		Args:
			pretrained (T.Union[str, int, bool]): If a string, the collection of pre-trained
			parameters of this name is returned. If an int, a collection of pre-trained
			parameters of this input size is returned. If True, a default collection
			of pre-trained parameters is returned. If False, the model's parameters are
			randomly initialized.
			Default is True.
			input_size (int): Input size the model expects. This argument is used only
			when pretrained is False.
			Default is 224.
			jit (bool): Whether to JIT the model's initialization function.
			Default is True.
			prng (T.Optional[jax.random.KeyArray]): PRNG key for
			initializing the model. If None, a key with a seed of 0 is created.
			Default is None.
			n_classes (int): Number of output classes. If 0, there is no head,
			and the raw final features are returned. If -1, all stages of the
			head, other than the final linear layer, are applied and the output
			returned.
			Default is 0.
			**model_kwargs: Additional arguments passed to the model.

		Returns (T.Union[T.Tuple[nn.Module, T.Dict], T.Tuple[nn.Module, T.Dict, T.Dict]]):
		The model, its parameters, and, if pretrained is not False, the normalization statistics
		associated with the pre-trained parameters.
		"""
		model = self.model_cls(
			**self.model_args,
			**model_kwargs,
			n_classes=n_classes,
			)
		vars = init_model(
			model=model,
			jit=jit,
			prng=prng,
			input_size=input_size,
			)

		if pretrained:
			pretrained_vars, norm_stats = self.get_params(
				pretrained=pretrained,
				n_classes=n_classes,
				)
			vars = merge_vars(vars, pretrained_vars)

		return (model, vars, norm_stats) if pretrained else (model, vars)
