"""
Constants used in deep learning:

- NORM_STATS: Normalization statistics.
"""


__all__ = [
	'NORM_STATS',
	]


NORM_STATS = {
	'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
	'inception': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)},
	'clip': {'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)}
	}
