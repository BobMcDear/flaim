"""
Tuplifies the input.
"""


__all__ = [
	'tuplify',
	]


import typing as T


def tuplify(
	item: T.Any,
	seq_len: int = 2,
	) -> T.Tuple:
	"""
	Tuplifies the input.

	Args:
		item (T.Any): Item to tuplify. If it
		is a T.Sequence, it is returned unchanged.
		Otherwise, a tuple consisting of seq_len repitions
		of item is returned.
		seq_len (int): How many times to repeat item
		when converting it to a tuple.
	
	Returns (T.Tuple): Tuplified version of item.
	"""
	return item if isinstance(item, T.Sequence) else seq_len*(item,)
