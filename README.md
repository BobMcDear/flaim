# Flax Image Models

• <strong>[Introduction](#introduction)</strong><br>
• <strong>[Installation](#installation)</strong><br>
• <strong>[Usage](#usage)</strong><br>
• <strong>[Available Architectures](#available-architectures)</strong><br>
• <strong>[Contributing](#contributing)</strong><br>
• <strong>[Acknowledgements](#acknowledgements)</strong><br>



## Introduction

flaim is a library of state-of-the-art pre-trained vision models, plus common deep learning modules in computer vision, for Flax.
It exposes a host of diverse image models through a straightforward interface with an emphasis on simplicity, leanness, and readability,
and offers lower-level modules for designing custom architectures.

## Installation

flaim can be installed through ```pip install flaim```. Beware that pip installs the CPU version of JAX, and you must [manually install JAX](https://github.com/google/jax#installation) yourself to run your programs on the GPU or TPU.

## Usage

```flaim.get_model``` is the central function of flaim and manages model retrieval. It accepts a handful
of arguments:
* ```model_name``` (```str```): The name of the model. If it is not recognized, an exception is thrown.
* ```pretrained``` (```bool```): Determines if pre-trained parameters are to be returned in lieu of randomly-initialized ones.
* ```n_classes``` (```int```): The number of output classes. This argument's value can fall under three groups:
  * 0: The model outputs the raw final feature maps. For instance, a ResNet is composed of a stem and four stages, followed
  by a head constituted of global average pooling and a fully-connected layer for generating predictions. When ```n_classes = 0```, the output of
  the fourth stage is returned, and the head is discarded. 
  * -1: Every part of the head, except for the linear layer, is applied and the output returned. In the ResNet example, the output of 
  the pooling layer is returned.
  * Positive integers: ```n_classes``` is interpreted as the desired number of output categories.
* ```jit``` (```bool```): Whether to JIT the model's initialization function. The benefit of JITting the initialization function 
is that no actual forward pass with real data is performed, unlike the default configuration. On the other hand, JIT compilation 
is generally a lengthy process.
* ```prng``` (```T.Optional[jax.random.KeyArray]```): PRNG key used for initializing the model. When ```None```,
a PRNG key, with a seed of 0, is created. If ```pretrained``` is ```True``` and ```n_classes``` is 0 or -1, this argument has no effects
on the returned parameters.
* ```norm_stats``` (```bool```): Whether to also return normalization statistics used to normalize the input data when the model was trained. The  statistics are returned as a dictionary, with key 'mean' containing the means and key 'std' the standard deviations for each channel.

The snippet below constructs a ResNet-50 with 10 output classes.

```python
import flaim


model, vars, norm_stats = flaim.get_model(
        model_name='resnet50',
        pretrained=True,
        n_classes=10,
        jit=True,
        prng=None,
        norm_stats=True,
        )
```

Performing a forward pass with flaim is similar to any other Flax module. However, networks
that behave differently during training versus inference, e.g., due to batch normalization, 
receive a ```training``` argument indicating whether the model should be in training mode or not. 

```python
from jax import numpy as jnp

# input should be normalized using norm_stats beforehand
input = jnp.ones((2, 224, 224, 3))

# Training
output, batch_stats = model.apply(
        vars,
        input,
        mutable=['batch_stats'],
        training=True,
        )

# Inference
output = model.apply(
        vars,
        input,
        training=False,
        )
```

Finally, intermediate activations can be captured by passing the string ```intermediates``` to ```mutable```. 

```python
output, batch_stats, intermediates = model.apply(
        vars,
        input,
        mutable=['batch_stats', 'intermediates'],
        training=True,
        )
```

If the model architecture is hierarchical, ```intermediates```'s items are the output of each stage and can be looked up through 
```intermediates['intermediates']['stage_ind']```, where ```ind``` is the index of the stage, with 0 being reserved for the stem. For isotropic models, the output of every block is returned, accessible via ```intermediates['intermediates']['block_ind']```.

Note that Flax's ```sow``` API, which is used to store the intermediate activations, appends the data to a tuple; that is, if _n_ forward passes are performed, ```intermediates['intermediates']['stage_ind']``` or ```intermediates['intermediates']['block_ind']``` would be tuples of length _n_, with the *i*<sup>th</sup> item corresponding to the *i*<sup>th</sup> forward pass.

## Available Architectures

All available architectures, accompanied by short descriptions and references, are [here](https://github.com/bobmcdear/flaim/blob/main/ARCHITECTURES.md). ```flaim.list_models``` also returns a list of flaim models. Its only arugment, ```pattern```, is an optional regex pattern that, if not ```None```, ensures only model names containing this expression are returned, as demonstrated below.

```python
# Every model
print(flaim.list_models())

# ResNeXt-based networks
print(flaim.list_models(r'resnext'))

# ViTs of resolution 224 x 224
print(flaim.list_models(r'vit.+224'))
```

## Contributing

Code contributions are currently not accepted, however, there are three alternatives for those interested in contributing to flaim: 

• Bugs: If you discover any bugs, please open an issue and include your system information, a description of the problem, and a reproducible example.<br>
• Feature request: flaim is under active development and many more models will be released in the near future. If there are particular architectures or modules you'd like to see added, please request them by opening an issue.<br>
• Questions: If you have questions regarding a model, the code, or anything else, please ask them by opening a discussion thread. Most likely somebody else has the same question, and asking it would help them too.<br>


## Acknowledgements

Many thanks to Ross Wightman for the amazing timm package, which was an inspiration for flaim and has been an indispensable guide during development. Additionally, the pre-trained parameters are stored on Hugging Face Hub; big thanks to Hugging Face for this gratis service.

References for ```flaim.models``` can be found [here](https://github.com/bobmcdear/flaim/blob/main/ARCHITECTURES.md), and ones for ```flaim.layers``` are in the source code.
