"""
Functions for accessing flaim models and loading pre-trained parameters.
"""


import os
import pickle
import re
import urllib.request
import typing as T

import jax
from flax import traverse_util
from flax import linen as nn
from jax import numpy as jnp


NORM_STATS = {
	'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
	'inception': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)},
	'clip': {'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)}
	}


def add_model_cls(
	model_cls: T.Callable,
	configs: T.Dict,
	) -> T.Dict:
	"""
	Adds the desired model class to every element in a
	dictionary of model configurations.

	Args:
		model_cls (T.Callable): Model class.
		configs (T.Dict): Dictionary of model configurations.
	
	Returns (T.Dict): Model configurations, with the model class
	added to them.
	"""
	for model_name in configs:
		configs[model_name]['model_cls'] = model_cls
	return configs


CONFIGS = {}
def register_configs(
	get_configs: T.Callable,
	) -> T.Callable:
	"""
	Decorator for registering model configurations
	by adding them to CONFIGS.

	Args:
		get_configs (T.Callable): T.Callable that returns 
		model configurations.
	
	Returns (T.Callable): get_configs unmodified.
	"""
	model_cls, configs = get_configs()
	configs = add_model_cls(model_cls, configs)
	CONFIGS.update(configs)
	return get_configs


def get_input_size(
	model_name: str,
	) -> int:
	"""
	Gets the input size a model expects.

	Args:
		model_name (str): Name of model.
	
	Returns (int): The input size the model expects.
	"""
	size = re.findall(r'_\d{3}(?:_|$)', model_name)
	return int(size[0].split('_')[1]) if size else 224


def create_flaim_dir() -> str:
	"""
	Creates a .flaim/ folder in the home directory if it doesn't
	already exist.

	Returns (str): Path to the .flaim/ directory
	"""
	home_dir = os.path.expanduser("~")
	flaim_dir = home_dir+'/.flaim/'

	if not os.path.exists(flaim_dir):
		os.mkdir(flaim_dir)
	
	return flaim_dir


def download_vars(
	model_name: str,
	) -> str:
	"""
	Downloads a model's parameters if it has not already been downloaded
	and saves it to the .flaim/ directory.

	Args:
		model_name (str): Name of model whose parameters are to be downloaded.
	
	Returns (str): Path to the parameters' file.
	"""
	flaim_dir = create_flaim_dir()
	params_path = flaim_dir+model_name
	
	if not os.path.exists(params_path):
		url = f'https://huggingface.co/BobMcDear/{model_name}/resolve/main/{model_name}.pickle'
		urllib.request.urlretrieve(
			url=url,
			filename=params_path,
			)
	
	return params_path


def load_pretrained_vars(
	model_name: str,
	) -> T.Dict:
	"""
	Loads a model's parameters from the .flaim/ directory. If it doesn't exist,
	it is downloaded.

	Args:
		model_name (str): Name of model whose parameters are returned.
	
	Returns (T.Dict): The model's parameters.
	"""
	params_path = download_vars(model_name)
	with open(params_path, 'rb') as f:
		params = pickle.load(f)
	return params


def init_model(
	model: nn.Module,
	jit: bool = True,
	prng: T.Optional[jax.random.KeyArray] = None,
	input_size: int = 224,
	):
	"""
	Initializes a model's parameters.

	Args:
		model (nn.Module): Model to initialize.
		jit (bool): Whether to JIT the initialization function.
		Default is True.
		prng (T.Optional[jax.random.KeyArray]): PRNG key for
		initializing the model. If None, a key is created.
		Default is None.
		input_size (int): The input size the model expects.
		Default is 224.
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
	
	Returns (T.Dict): Merged dictionary of vars and pretrained_vars.
	"""
	vars = traverse_util.flatten_dict(vars)
	pretrained_vars = traverse_util.flatten_dict(pretrained_vars)

	merged_vars = {
		**vars,
		**pretrained_vars,
		} 
	merged_vars = traverse_util.unflatten_dict(merged_vars)

	return merged_vars


def get_model(
	model_name: str,
	pretrained: bool = True,
	n_classes: int = 0,
	jit: bool = True,
	prng: T.Optional[jax.random.KeyArray] = None,
	norm_stats: bool = False,
	) -> T.Union[T.Tuple[nn.Module, T.Dict], T.Tuple[nn.Module, T.Dict, T.Dict]]:
	"""
	Returns a model and its parameters.

	Args:
		model_name (str): Name of model.
		pretrained (bool): Whether to return pre-trained parameters.
		If False, the parameters are randomly initialized.
		Default is True.
		n_classes (int): Number of output classes. If 0, there is no head,
		and the raw final features are returned. If -1, all stages of the 
		head, other than the final linear layer, are applied and the output
		returned.
		Default is 0.
		jit (bool): Whether to JIT the initialization function.
		Default is True.
		prng (T.Optional[jax.random.KeyArray]): PRNG key for
		initializing the model's parameters. If None, a key is created.
		Default is None.
		norm_stats (bool): Whether to also return normalization statistics.
		The statistics are returned as a dictionary, with key 'mean' containing
		the means and key 'std' the standard deviations.
		Default is False.

	Returns (T.Union[T.Tuple[nn.Module, T.Dict], T.Tuple[nn.Module, T.Dict, T.Dict]]):
	The model, its parameters, and possibly the normalization statistics.
	"""
	if model_name not in CONFIGS:
		raise ValueError(f'{model_name} is not a recognized model.')

	config = CONFIGS[model_name]
	config['n_classes'] = n_classes

	model_cls = config.pop('model_cls')
	norm_stats_ = config.pop('norm_stats') if 'norm_stats' in config else NORM_STATS['imagenet']

	model = model_cls(**config)
	config['model_cls'] = model_cls
	config['norm_stats'] = norm_stats_

	vars = init_model(
		model=model,
		jit=jit,
		prng=prng,
		input_size=get_input_size(model_name),
		)
	
	if pretrained:
		pretrained_vars = load_pretrained_vars(model_name)
		vars = merge_vars(vars, pretrained_vars)
	
	return (model, vars, norm_stats_) if norm_stats else (model, vars)


def list_models(
	pattern: T.Optional[str] = None,
	) -> T.List:
	"""
	Lists available models.

	Args:
		pattern (T.Optional[str]): If not None, only the names of
		the models that contain this regex pattern are returned.
		Default is None.

	Returns (T.List): List of names of all available models.
	"""
	names = list(CONFIGS)
	return [name for name in names if bool(re.search(pattern, name))] if pattern else names
