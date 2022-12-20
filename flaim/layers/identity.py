"""
Identities:
- identity: Identity (function).
- Identity: Identity (class).
"""


__all__ = [
	'identity',
	'Identity',
	]


from flax import linen as nn


def identity(input, *args, **kwargs):
	"""
	Identity function.

	Args:
		input: Input.
		*args.
		**kwargs.
	
	Returns: Input.
	"""
	return input


class Identity(nn.Module):
	"""
	Identity class.
	"""
	@nn.compact
	def __call__(self, input, *args, **kwargs):
		return input
