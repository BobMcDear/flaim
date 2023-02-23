"""
The flaim hub, responsible for maintaining and retrieving pre-trained
parameters.
"""


import os
import pickle
import urllib.request
import typing as T
from functools import partial

from flax import traverse_util

from ..layers import NORM_STATS


def create_hub() -> str:
	"""
	Creates a folder for the flaim hub in the home directory.

	Returns (str): Path to the flaim hub.
	"""
	home_dir = os.path.expanduser('~')
	flaim_dir = home_dir+'/.flaim/'

	if not os.path.exists(flaim_dir):
		os.mkdir(flaim_dir)

	return flaim_dir


def download_params(
	repo_name: str,
	) -> str:
	"""
	Downloads a set of pre-trained parameters from Hugging Face Hub
	and stores it in the flaim hub.

	Args:
		repo_name (str): Name of Hugging Face Hub repository
		containing the pre-trained parameters.

	Returns (str): Path to the downloaded pre-trained parameters.
	"""
	flaim_dir = create_hub()
	params_path = flaim_dir+repo_name

	if not os.path.exists(params_path):
		urllib.request.urlretrieve(
			url=f'https://huggingface.co/BobMcDear/{repo_name}/resolve/main/{repo_name}.pickle',
			filename=params_path,
			)

	return params_path


def remove_head(
	params: T.Dict,
	) -> T.Dict:
	"""
	Removes the head's parameters from a dictionary of parameters.

	Args:
		params (T.Dict): Dictionary of parameters to remove the
		head from.

	Returns (T.Dict): The parameters with the head removed.
	"""
	flattened_params = traverse_util.flatten_dict(params)
	flattened_params = {key: flattened_params[key] for key in flattened_params if 'Head' not in key[1]}
	return traverse_util.unflatten_dict(flattened_params)


def load_params(
	repo_name: str,
	n_classes: int = 0,
	) -> T.Dict:
	"""
	Downloads and returns a model's pre-trained parameters.

	Args:
		repo_name (str): Name of Hugging Face Hub repository
		containing the pre-trained parameters.
		n_classes (int): Number of output classes. If 0, the head's parameters
		are removed.

	Returns (T.Dict): The model's parameters.
	"""
	params_path = download_params(repo_name)
	with open(params_path, 'rb') as f:
		params = pickle.load(f)

	if n_classes == 0:
		params = remove_head(params)

	return params


def params_config(
	repo_name: str,
	norm_stats: T.Dict,
	) -> T.Dict:
	"""
	Gets a configuration dictionary for pre-trained parameters.

	Args:
		repo_name (str): Name of Hugging Face Hub repository
		containing the pre-trained parameters.
		norm_stats (T.Dict): Normalization statistics associated
		with the pre-trained parameters.

	Returns (T.Dict): Dictionary containing the repository name
	and normalization statistics.
	"""
	return {
		'repo_name': repo_name,
		'norm_stats': norm_stats,
		}
imagenet_params_config = partial(params_config, norm_stats=NORM_STATS['imagenet'])
inception_params_config = partial(params_config, norm_stats=NORM_STATS['inception'])
clip_params_config = partial(params_config, norm_stats=NORM_STATS['clip'])
