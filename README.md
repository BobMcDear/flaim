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
and supplies lower-level modules for designing custom architectures.

## Installation

flaim can be installed through ```pip install flaim```. Beware that pip installs the CPU version of JAX, and you must [manually install JAX](https://github.com/google/jax#installation) yourself to run your programs on a GPU or TPU.

## Usage

```flaim.get_model``` is the central function of flaim and manages model retrieval. It takes a handful
of arguments:
* ```model_name``` (```str```): The name of the desired model.
* ```pretrained``` (```Union[str, int, bool]```): Every flaim network is accompanied by at least one group of pre-trained
parameters. For example, those of MaxViT-Small (```maxvit_small```) are ```in1k_224```, ```in1k_384```, and ```in1k_512```,
corresponding to parameters trained on ImageNet1K at resolutions 224 x 224, 384 x 384, and 512 x 512 respectively. When ```pretrained``` is
  * A string, the pre-trained parameters of this name are returned, e.g., ```pretrained = 'in1k_224'```. This is the recommended means of loading pre-trained parameters, for it is the most explicit.
  * An integer, a set of parameters trained at this resolution is returned. For instance, ```pretrained = 384``` would return a set of parameters
  trained at a resolution of 384 x 384. It should be borne in mind that some models might not have parameters trained at this resolution, in which case an exception is thrown.
  * ```True```, a default collection of pre-trained parameters is returned. Users should be wary of this option because certain models such as MaxViT cannot handle variable resolutions, and therefore
  the returned pre-trained parameters might not be compatible with one's input shapes. In such scenarios, passing the desired resolution to ```pretrained``` would be the more judicious choice.
  * ```False```, the parameters are randomly-initialized.
<br><br>

* ```input_size``` (```int```): When ```pretrained``` is ```False```, ```input_size``` refers to the input size the model should expect
and is used to initialize the parameters. Providing the correct value for ```input_size``` is especially important for fixed-resolution
architectures such as ViT.
* ```jit``` (```bool```): Whether to JIT the model's initialization function. The advantage of JITting the initialization function
is that no actual forward pass with real data is performed, unlike the default configuration. On the other hand, JIT compilation
can be a time-consuming process.
* ```prng``` (```Optional[jax.random.KeyArray]```): PRNG key used for initializing the model. A PRNG key with a seed of 0 is created when ```prng = None```.
* ```n_classes``` (```int```): The number of output classes. This argument's value can fall under three groups:
  * 0: The model outputs the raw final feature maps. For instance, a ResNet is composed of a stem and four stages, followed
  by a head constituted of global average pooling and a fully-connected layer. When the value of this argument is 0, the output of
  the fourth stage is returned, and the head is discarded.
  * -1: Every part of the head, except for the linear layer, is applied and the output returned. In the ResNet example, the output of
  the pooling layer is returned.
  * Positive integers: ```n_classes``` is interpreted as the desired number of output categories.
<br><br>

```flaim.get_model``` returns the model, its parameters, and, if ```pretrained``` is not ```False```, the normalization statistics associated with the pre-trained parameters. The snippet below constructs an ImageNet1K-trained ResNet-50 with 10 output classes.

```python
import flaim


model, vars, norm_stats = flaim.get_model(
        model_name='resnet50',
        pretrained='in1k_224',
        n_classes=10,
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
output, new_batch_stats = model.apply(vars, input, training=True, mutable=['batch_stats'])
# Inference
output = model.apply(vars, input, training=False, mutable=False)
```

Finally, the model's intermediate activations can be captured by passing ```intermediates``` to ```mutable```.

```python
output, intermediates = model.apply(vars, input, training=False, mutable=['intermediates'])
```

If the model is hierarchical, ```intermediates```'s entries are the output of each network stage and can be looked up through
```intermediates['intermediates']['stage_ind']```, where ```ind``` is the index of the desired stage, with 0 being reserved for the stem. For isotropic models, the output of every block is returned, accessible via ```intermediates['intermediates']['block_ind']```, where ```ind``` is the index of the desired block and 0 is once again reserved for the stem.

It should be noted that Flax's [```sow```](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.Module.sow) API, which is used utilized by flaim, appends the intermediate activations to a tuple; that is, if _n_ forward passes are performed, ```intermediates['intermediates']['stage_ind']``` or ```intermediates['intermediates']['block_ind']``` would be tuples of length _n_, with the *i*<sup>th</sup> item corresponding to the *i*<sup>th</sup> forward pass.

## Examples

[```examples/```](https://github.com/bobmcdear/flaim/blob/main/examples/) includes a series of annotated notebooks for solving various vision problems such as object classification using flaim.

## Available Architectures

All available architectures and their pre-trained parameters, plus short descriptions and references, are listed [here](https://github.com/bobmcdear/flaim/blob/main/ARCHITECTURES.md).

```flaim.list_models``` also returns a list of (name of model, name of pre-trained parameters) pairs, e.g., (```resnet50```, ```in1k_224```) and has two arguments:

* ```model_pattern``` (```str```): A regex pattern that, if not an empty string, ensures only pairs where the model name contains this expression are returned.
* ```params_pattern``` (```Union[str, int]```): If ```params_pattern``` is a non-empty string, only pairs where the pre-trained parameters' name contains this regex pattern are returned. When an integer, only pairs where the pre-trained parameters were trained on images of this resolution are returned.

This function is demonstrated below.

```python
# Every model
print(flaim.list_models())

# ResNeXt-based networks of depth 50
print(flaim.list_models(model_pattern='resnext50'))

# Models trained on ImageNet22K
print(flaim.list_models(params_pattern='in22k'))

# ViTs of input size 384 x 384
print(flaim.list_models(model_pattern='^vit', params_pattern=384))
```

## Contributing

Code contributions are currently not accepted, however, there are three alternatives for those seeking to help flaim evolve:

* Bugs: If you discover any bugs, please [open up an issue](https://github.com/BobMcDear/flaim/issues/new?assignees=BobMcDear&labels=bug&template=bug_report.md&title=%5BBug+report%5D), specify your system information, and provide a description of the problem as well as a reproducible example.<br>
* Feature request: If there are particular architectures or modules that you think would be beneficial additions to flaim, please request them in an [Ideas discussion thread](https://github.com/BobMcDear/flaim/discussions/new?category=ideas).<br>
* Questions: If you have questions regarding a model, a segment of code, or anything else, please ask them by creating a [Q&A discussion thread](https://github.com/BobMcDear/flaim/discussions/new?category=q-a).<br>


## Acknowledgements

Many thanks to Ross Wightman for the amazing timm package, which was an inspiration for flaim and has been an indispensable guide during development. Additionally, the pre-trained parameters are stored on Hugging Face Hub; big thanks to Hugging Face for offering this service gratis.

References for ```flaim.models``` can be found [here](https://github.com/bobmcdear/flaim/blob/main/ARCHITECTURES.md), and ones for ```flaim.layers``` are in the source code.
