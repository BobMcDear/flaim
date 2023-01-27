from setuptools import find_packages, setup


setup(
	name='flaim',
	version='0.0.5',
	description='Flax Image Models',
	long_description=open('README.md').read(),
	long_description_content_type='text/markdown',
	author='Borna Ahmadzadeh',
	author_email='borna.ahz@gmail.com',
	url='https://github.com/bobmcdear/flaim',
	packages=find_packages(),
	license='GNU',
	keywords=[
		'computer vision',
		'machine learning',
		'deep learning',
		'jax',
		'flax',
		],
	install_requires=[
		'jaxlib',
		'numpy>=1.20',
		'jax>=0.3.23',
		'flax>=0.6.0'
		],
	python_requires='>=3.8'
	)
