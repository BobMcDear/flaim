# Flax Image Models

• <strong>[Introduction](#introduction)</strong><br>
• <strong>[Installation](#installation)</strong><br>
• <strong>[Usage](#usage)</strong><br>
• <strong>[Examples](#examples)</strong><br>
• <strong>[Available Architectures](#available-architectures)</strong><br>
• <strong>[Contributing](#contributing)</strong><br>
• <strong>[Acknowledgements](#acknowledgements)</strong><br>



## Introduction

flaim is a library of state-of-the-art pre-trained vision models, plus common deep learning modules in computer vision, for Flax.
It exposes a host of diverse image models through a straightforward interface with an emphasis on simplicity, leanness, and readability,
and offers lower-level modules for designing custom architectures.

## Installation

flaim can be installed through ```pip install flaim```. Beware that pip installs the CPU version of JAX, and you must [manually install JAX](https://github.com/google/jax#installation) yourself to run your programs on a GPU or TPU.

## Usage

```flaim.get_model``` is the central function of flaim and manages model retrieval. It accepts a handful
of arguments:
* ```model_name``` (```str```): The name of the desired model. If it is not recognized, an exception is thrown.
* ```pretrained``` (```bool```): Determines if pre-trained parameters are to be returned in lieu of randomly-initialized ones.
* ```n_classes``` (```int```): The number of output classes. This argument's value can fall under three groups:
  * 0: The model outputs the raw final feature maps. For instance, a ResNet is composed of a stem and four stages, followed
  by a head constituted of global average pooling and a fully-connected layer. When ```n_classes = 0```, the output of
  the fourth stage is returned, and the head is discarded. 
  * -1: Every part of the head, except for the linear layer, is applied and the output returned. In the ResNet example, the output of 
  the pooling layer is returned.
  * Positive integers: ```n_classes``` is interpreted as the desired number of output categories.
* ```jit``` (```bool```): Whether to JIT the model's initialization function. The advantage of JITting the initialization function 
is that no actual forward pass with real data is performed, unlike the default configuration. On the other hand, JIT compilation 
can be a time-consuming process.
* ```prng``` (```Optional[jax.random.KeyArray]```): PRNG key used for initializing the model. When ```None```,
a PRNG key, with a seed of 0, is created. If ```pretrained = True``` and ```n_classes``` is 0 or -1, this argument has no effects on the model parameters.
* ```norm_stats``` (```bool```): Whether to also return the normalization statistics used to normalize the input data during training. These statistics are returned as a dictionary, with key ```mean``` holding the means and key ```std``` the standard deviations for each channel.

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

Performing a forward pass with flaim is similar to any other Flax model. However, networks
that behave differently during training versus inference, e.g., due to batch normalization, 
receive a ```training``` argument indicating whether the model should be in training mode or not. Furthermore, like
any other Flax module incorporating batch normalization, ```batch_stats``` must be passed to ```mutable```
to update batch normalization's running statistics during training.

```python
from jax import numpy as jnp

# input should be normalized using norm_stats beforehand
input = jnp.ones((2, 224, 224, 3))

# Training
output, new_batch_stats = model.apply(
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

Finally, the model's intermediate activations can be captured by passing ```intermediates``` to ```mutable```. 

```python
output, new_batch_stats, intermediates = model.apply(
        vars,
        input,
        mutable=['batch_stats', 'intermediates'],
        training=True,
        )
```

If the model architecture is hierarchical, ```intermediates```'s entries are the output of each network stage and can be looked up through 
```intermediates['intermediates']['stage_ind']```, where ```ind``` is the index of the desired stage, with 0 being reserved for the stem. For isotropic models, the output of every block is returned, accessible via ```intermediates['intermediates']['block_ind']```, where ```ind``` is the index of the desired block and 0 is once again reserved for the stem. 

Bear in mind that Flax's [```sow```](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.sow) API, which is used to store the intermediate activations, appends the data to a tuple; that is, if _n_ forward passes are performed, ```intermediates['intermediates']['stage_ind']``` or ```intermediates['intermediates']['block_ind']``` would be tuples of length _n_, with the *i*<sup>th</sup> item corresponding to the *i*<sup>th</sup> forward pass.

## Examples

[```examples/```](https://github.com/bobmcdear/flaim/blob/main/examples/) includes a series of annotated notebooks for solving various vision problems such as object classification using flaim.  

## Available Architectures

All available architectures, accompanied by short descriptions and references, are listed [here](https://github.com/bobmcdear/flaim/blob/main/ARCHITECTURES.md). ```flaim.list_models``` also returns a list of flaim models, and its only arugment, ```pattern```, is an optional regex pattern that, if not ```None```, ensures only model names containing this expression are returned, as demonstrated below.

```python
# Every model
print(flaim.list_models())

# ResNeXt-based networks
print(flaim.list_models(r'resnext'))

# ViTs of resolution 224 x 224
print(flaim.list_models(r'vit.+224'))
```

## Contributing

Code contributions are currently not accepted, however, there are three alternatives for those interested in helping flaim evolve: 

• Bugs: If you discover any bugs, please [open up an issue](https://github.com/BobMcDear/flaim/issues/new?assignees=BobMcDear&labels=bug&template=bug_report.md&title=%5BBug+report%5D), specify your system information, and provide a description of the problem as well as a reproducible example.<br>
• Feature request: If there are particular architectures or modules that you think would be beneficial additions to flaim, please request them in an [Ideas discussion thread](https://github.com/BobMcDear/flaim/discussions/new?category=ideas).<br>
• Questions: If you have questions regarding a model, a segment of code, or anything else, please ask them by creating a [Q&A discussion thread](https://github.com/BobMcDear/flaim/discussions/new?category=q-a).<br>


## Acknowledgements

Many thanks to Ross Wightman for the amazing timm package, which was an inspiration for flaim and has been an indispensable guide during development. Additionally, the pre-trained parameters are stored on Hugging Face Hub; big thanks to Hugging Face for this gratis service.

References for ```flaim.models``` can be found [here](https://github.com/bobmcdear/flaim/blob/main/ARCHITECTURES.md), and ones for ```flaim.layers``` are in the source code.
