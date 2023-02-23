"""
Factory for accessing flaim models and pre-trained parameters.
"""


from .hub import clip_params_config, imagenet_params_config, inception_params_config
from .registry import get_model, list_models, register_models
