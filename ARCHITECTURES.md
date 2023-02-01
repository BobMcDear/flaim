# Available Architectures

The following is a list of all available architectures, with brief descriptions and references, classified into network families. Unless otherwise noted, the training dataset is ImageNet1K.

• <strong>[CaiT](#cait)</strong><br>
• <strong>[ConvMixer](#convmixer)</strong><br>
• <strong>[ConvNeXt](#convnext)</strong><br>
• <strong>[ConvNeXt V2](#convnext-v2)</strong><br>
• <strong>[CLIP](#clip)</strong><br>
• <strong>[EfficientNetV2](#efficientnetv2)</strong><br>
• <strong>[HorNet](#hornet)</strong><br>
• <strong>[MaxViT](#maxvit)</strong><br>
• <strong>[GC ViT](#gc-vit)</strong><br>
• <strong>[NesT](#nest)</strong><br>
• <strong>[PVT V2](#pvt-v2)</strong><br>
• <strong>[ResNet](#resnet)</strong><br>
• <strong>[ResNet-D](#resnet-d)</strong><br>
• <strong>[ResNet-T](#resnet-t)</strong><br>
• <strong>[Wide ResNet](#wide-resnet)</strong><br>
• <strong>[ResNeXt](#resnext)</strong><br>
• <strong>[RegNet](#regnet)</strong><br>
• <strong>[SENet](#senet)</strong><br>
• <strong>[ECANet](#ecanet)</strong><br>
• <strong>[ResNet-RS](#resnet-rs)</strong><br>
• <strong>[SKNet](#sknet)</strong><br>
• <strong>[ResNeSt](#resnest)</strong><br>
• <strong>[Swin](#swin)</strong><br>
• <strong>[Swin-S3](#swin-s3)</strong><br>
• <strong>[VAN](#van)</strong><br>
• <strong>[VGG](#vgg)</strong><br>
• <strong>[ViT](#vit)</strong><br>
• <strong>[ViT SAM](#vit-sam)</strong><br>
• <strong>[ViT DINO](#vit-dino)</strong><br>
• <strong>[DeiT 3](#deit-3)</strong><br>
• <strong>[BEiT](#beit)</strong><br>
• <strong>[BEiT V2](#beit-v2)</strong><br>
• <strong>[XCiT](#xcit)</strong><br>


## CaiT

Class attention image transformer (CaiT) from _[Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)_ by Touvron et al. CaiT
presents two novel modules, LayerScale and class attention, for enabling ViTs to go significantly deeper with little saturation in accuracy
at greater depths.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/cait.py).

• ```cait_xxsmall24_224```: CaiT-XXSmall, depth 24, resolution 224 x 224.

• ```cait_xxsmall36_224```: CaiT-XXSmall, depth 36, resolution 224 x 224.

• ```cait_small24_224```: CaiT-Small, depth 24, resolution 224 x 224.

• ```cait_xxsmall24_384```: CaiT-XXSmall, depth 24, resolution 384 x 384.

• ```cait_xxsmall36_384```: CaiT-XXSmall, depth 36, resolution 384 x 384.

• ```cait_xsmall24_384```: CaiT-XSmall, depth 24, resolution 384 x 384.

• ```cait_small24_384```: CaiT-Small, depth 24, resolution 384 x 384.

• ```cait_small36_384```: CaiT-Small, depth 36, resolution 384 x 384.

• ```cait_medium36_384```: CaiT-Medium, depth 36, resolution 384 x 384.

• ```cait_medium48_448```: CaiT-Medium, depth 48, resolution 448 x 448.

## ConvMixer

ConvMixer from _[Patches Are All You Need?](https://arxiv.org/abs/2201.09792)_ by Trockman et al. ConvMixer
is similar to isotropic architectures like ViT but uses convolutions with large kernel sizes to perform token mixing.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convmixer.py).

• ```convmixer20_1024d_patch14_kernel9```: ConvMixer, depth 20, token dimension 1024, patch size 14 x 14, kernel size 9 x 9.

• ```convmixer20_1536d_patch7_kernel9```: ConvMixer, depth 20, token dimension 1536, patch size 7 x 7, kernel size 9 x 9.

• ```convmixer32_768d_patch7_kernel7```: ConvMixer, depth 32, token dimension 768, patch size 7 x 7, kernel size 7 x 7.


## ConvNeXt

ConvNeXt from _[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)_ by Liu et al. ConvNeXt borrows ideas from the transformer literature, such as larger kernel sizes and more aggressive downsampling in the stem, to modernize a plain ResNet and attain results on par with state-of-the-art
vision transformers like Swin using a purely convolutional network. 
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convnext.py).

• ```convnext_xxxnano```: ConvNeXt-XXXNano, corresponding to ConvNeXt-Atto in timm.

• ```convnext_xxnano```: ConvNeXt-XXNano, corresponding to ConvNeXt-Femto in timm.

• ```convnext_xnano```: ConvNeXt-XNano, corresponding to ConvNeXt-Pico in timm.

• ```convnext_nano```: ConvNeXt-Nano.

• ```convnext_tiny```: ConvNeXt-Tiny.

• ```convnext_small```: ConvNeXt-Small.

• ```convnext_base```: ConvNeXt-Base.

• ```convnext_large```: ConvNeXt-Large.

• ```convnext_nano_in12k```: ConvNeXt-Nano, trained on ImageNet12K (a subset of ImageNet22K).

• ```convnext_tiny_in12k```: ConvNeXt-Tiny, trained on ImageNet12K (a subset of ImageNet22K).

• ```convnext_small_in12k```: ConvNeXt-Small, trained on ImageNet12K (a subset of ImageNet22K).

• ```convnext_nano_in12ft1k```: ConvNeXt-Nano, pre-trained on ImageNet12K (a subset of ImageNet22K) and fine-tuned on ImageNet1K.

• ```convnext_tiny_in12ft1k```: ConvNeXt-Tiny, pre-trained on ImageNet12K (a subset of ImageNet22K) and fine-tuned on ImageNet1K.

• ```convnext_small_in12ft1k```: ConvNeXt-Small, pre-trained on ImageNet12K (a subset of ImageNet22K) and fine-tuned on ImageNet1K.

• ```convnext_tiny_384_in12ft1k```: ConvNeXt-Tiny, training resolution 384 x 384, pre-trained on ImageNet12K (a subset of ImageNet22K) and fine-tuned on ImageNet1K.

• ```convnext_small_384_in12ft1k```: ConvNeXt-Small, training resolution 384 x 384, pre-trained on ImageNet12K (a subset of ImageNet22K) and fine-tuned on ImageNet1K.

• ```convnext_tiny_in22k```: ConvNeXt-Tiny, trained on ImageNet22K.

• ```convnext_small_in22k```: ConvNeXt-Small, trained on ImageNet22K.

• ```convnext_base_in22k```: ConvNeXt-Base, trained on ImageNet22K.

• ```convnext_large_in22k```: ConvNeXt-Large, trained on ImageNet22K.

• ```convnext_xlarge_in22k```: ConvNeXt-XLarge, trained on ImageNet22K.

• ```convnext_tiny_in22ft1k```: ConvNeXt-Tiny, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_small_in22ft1k```: ConvNeXt-Small, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_base_in22ft1k```: ConvNeXt-Base, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_large_in22ft1k```: ConvNeXt-Large, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_xlarge_in22ft1k```: ConvNeXt-XLarge, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_tiny_384_in22ft1k```: ConvNeXt-Tiny, training resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_small_384_in22ft1k```: ConvNeXt-Small, training resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_base_384_in22ft1k```: ConvNeXt-Base, training resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_large_384_in22ft1k```: ConvNeXt-Large, training resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```convnext_xlarge_384_in22ft1k```: ConvNeXt-XLarge, training resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

## ConvNeXt V2

ConvNeXt V2 from _[ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)_ by Woo et al. ConvNeXt V2 is based on ConvNeXt and benefits from a fully convolutional masked autoencoder training scheme (FCMAE) as well as global response normalization (GRN), a novel module that abates inter-channel feature redundancies.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convnext.py).

• ```convnextv2_atto_fcmae```: ConvNeXtV2-Atto, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_femto_fcmae```: ConvNeXtV2-Femto, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_pico_fcmae```: ConvNeXtV2-Pico, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_nano_fcmae```: ConvNeXtV2-Nano, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_tiny_fcmae```: ConvNeXtV2-Tiny, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_base_fcmae```: ConvNeXtV2-Base, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_large_fcmae```: ConvNeXtV2-Large, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_huge_fcmae```: ConvNeXtV2-Huge, trained using FCMAE on ImageNet1K with no supervision.

• ```convnextv2_atto_fcmae_ftin1k```: ConvNeXtV2-Atto, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_femto_fcmae_ftin1k```: ConvNeXtV2-Femto, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_pico_fcmae_ftin1k```: ConvNeXtV2-Pico, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_nano_fcmae_ftin1k```: ConvNeXtV2-Nano, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_tiny_fcmae_ftin1k```: ConvNeXtV2-Tiny, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_base_fcmae_ftin1k```: ConvNeXtV2-Base, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_large_fcmae_ftin1k```: ConvNeXtV2-Large, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_huge_fcmae_ftin1k```: ConvNeXtV2-Huge, pre-trained using FCMAE on ImageNet1K and fine-tuned with labels.

• ```convnextv2_nano_fcmae_in22ft1k```: ConvNeXtV2-Nano, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_tiny_fcmae_in22ft1k```: ConvNeXtV2-Tiny, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_base_fcmae_in22ft1k```: ConvNeXtV2-Base, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_large_fcmae_in22ft1k```: ConvNeXtV2-Large, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_nano_384_fcmae_in22ft1k```: ConvNeXtV2-Nano, training resolution 384 x 384, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_tiny_384_fcmae_in22ft1k```: ConvNeXtV2-Tiny, training resolution 384 x 384, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_base_384_fcmae_in22ft1k```: ConvNeXtV2-Base, training resolution 384 x 384, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_large_384_fcmae_in22ft1k```: ConvNeXtV2-Large, training resolution 384 x 384, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_huge_384_fcmae_in22ft1k```: ConvNeXtV2-Huge, training resolution 384 x 384, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

• ```convnextv2_huge_512_fcmae_in22ft1k```: ConvNeXtV2-Huge, training resolution 512 x 512, pre-trained using FCMAE on ImageNet22K and fine-tuned on ImageNet1K with labels.

## CLIP

Models trained using contrastive language-image pre-training (CLIP) from _[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)_ by Radford et al. CLIP jointly trains a vision and language model (flaim includes the former only) on (image, text) pairs, with the objective of ensuring the two models' output features are close for matching (image, text) pairs and distant otherwise.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convnext.py) and [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py).

• ```convnext_base_clip_laion2b```: ConvNeXt-Base, trained on LAION-2B.

• ```convnext_base_clip_laion2b_augreg```: ConvNeXt-Base, trained on LAION-2B with additional augmentation and regularization.

• ```convnext_base_clip_laiona```: ConvNeXt-Base, trained on LAION-Aesthetics.

• ```convnext_base_clip_320_laiona```: ConvNeXt-Base, training resolution 320 x 320, trained on LAION-Aesthetics.

• ```convnext_base_clip_320_laiona_augreg```: ConvNeXt-Base, training resolution 320 x 320, trained on LAION-Aesthetics with additional augmentation and regularization.

```vit_base_clip_patch32_224_laion2b```: ViT-Base, patch size 32 x 32, resolution 224 x 224, trained on LAION-2B.

```vit_base_clip_patch16_224_laion2b```: ViT-Base, patch size 16 x 16, resolution 224 x 224, trained on LAION-2B.

```vit_large_clip_patch14_224_laion2b```: ViT-Large, patch size 14 x 14, resolution 224 x 224, trained on LAION-2B.

```vit_huge_clip_patch14_224_laion2b```: ViT-Huge, patch size 14 x 14, resolution 224 x 224, trained on LAION-2B.

```vit_giant_clip_patch14_224_laion2b```: ViT-Giant, patch size 14 x 14, resolution 224 x 224, trained on LAION-2B.

```vit_base_clip_patch32_224_openai```: ViT-Base, patch size 32 x 32, resolution 224 x 224, trained by OpenAI on a dataset of 400 million pairs.

```vit_base_clip_patch16_224_openai```: ViT-Base, patch size 16 x 16, resolution 224 x 224, trained by OpenAI on a dataset of 400 million pairs.

```vit_large_clip_patch14_224_openai```: ViT-Large, patch size 14 x 14, resolution 224 x 224, trained by OpenAI on a dataset of 400 million pairs.

## EfficientNetV2

EfficientNetV2 from _[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)_ by Tan et al. EfficientNetV2 
builds on EfficientNet but adds fused MBConv to the search space, takes training speed into account, and more.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/efficientnet.py).

• ```efficientnetv2_small```: EfficientNetV2-Small.

• ```efficientnetv2_medium```: EfficientNetV2-Medium.

• ```efficientnetv2_large```: EfficientNetV2-Large.

• ```efficientnetv2_small_in22k```: EfficientNetV2-Small, trained on ImageNet22K.

• ```efficientnetv2_medium_in22k```: EfficientNetV2-Medium, trained on ImageNet22K.

• ```efficientnetv2_large_in22k```: EfficientNetV2-Large, trained on ImageNet22K.

• ```efficientnetv2_xlarge_in22k```: EfficientNetV2-XLarge, trained on ImageNet22K.

• ```efficientnetv2_small_in22ft1k```: EfficientNetV2-Small, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```efficientnetv2_medium_in22ft1k```: EfficientNetV2-Medium, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```efficientnetv2_large_in22ft1k```: EfficientNetV2-Large, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

• ```efficientnetv2_xlarge_in22ft1k```: EfficientNetV2-XLarge, pre-trained on ImageNet22K and fine-tuned on ImageNet1K.

## HorNet

High-order spatial interaction network (HorNet) from _[HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/abs/2207.14284)_ by Rao et al. HorNet uses recursive gated convolutions (g^n convolution) to model long-range spatial interactions in the input, akin to self-attention, via convolutions.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/raoyongming/HorNet).

• ```hornet_tiny```: HorNet-Tiny, kernel size 7 x 7.

• ```hornet_small```: HorNet-Small, kernel size 7 x 7.

• ```hornet_base```: HorNet-Base, kernel size 7 x 7.

• ```hornet_large_in22k```: HorNet-Large, kernel size 7 x 7, trained on ImageNet22K.

## MaxViT

Multi-axis vision transformer (MaxViT) from _[MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)_ by Tu et al. MaxViT combines
dilated global attention, a method for expressing global spatial interactions with a linear complexity, with MBConv and window attention for competitive performance.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/maxxvit.py).

```maxvit_tiny_224```: MaxViT-Tiny, resolution 224 x 224.

```maxvit_small_224```: MaxViT-Small, resolution 224 x 224.

```maxvit_base_224```: MaxViT-Base, resolution 224 x 224.

```maxvit_large_224```: MaxViT-Large, resolution 224 x 224.

```maxvit_tiny_384```: MaxViT-Tiny, resolution 384 x 384.

```maxvit_small_384```: MaxViT-Small, resolution 384 x 384.

```maxvit_base_384```: MaxViT-Base, resolution 384 x 384.

```maxvit_large_384```: MaxViT-Large, resolution 384 x 384.

```maxvit_tiny_512```: MaxViT-Tiny, resolution 512 x 512.

```maxvit_small_512```: MaxViT-Small, resolution 512 x 512.

```maxvit_base_512```: MaxViT-Base, resolution 512 x 512.

```maxvit_large_512```: MaxViT-Large, resolution 512 x 512.

```maxvit_base_384_in22ft1k```: MaxViT-Base, resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

```maxvit_large_384_in22ft1k```: MaxViT-Large, resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

```maxvit_xlarge_384_in22ft1k```: MaxViT-XLarge, resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

```maxvit_base_512_in22ft1k```: MaxViT-Base, resolution 512 x 512, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

```maxvit_large_512_in22ft1k```: MaxViT-Large, resolution 512 x 512, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

```maxvit_xlarge_512_in22ft1k```: MaxViT-XLarge, resolution 512 x 512, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

## GC ViT

Global context vision transformer (GC ViT) from _[Global Context Vision Transformers](https://arxiv.org/abs/2206.09959)_ by Hatamizadeh et al. GC ViT's core contribution is global context attention, where global queries are matched against local keys and values for capturing long-range interactions,
in addition to short-range interactions modelled via window attention.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/gcvit.py).

• ```gcvit_xxtiny_224```: GCViT-XXTiny, resolution 224 x 224.

• ```gcvit_xtiny_224```: GCViT-XTiny, resolution 224 x 224.

• ```gcvit_tiny_224```: GCViT-Tiny, resolution 224 x 224.

• ```gcvit_small_224```: GCViT-Small, resolution 224 x 224.

• ```gcvit_base_224```: GCViT-Base, resolution 224 x 224.

## NesT

Nested transformer (NesT) from _[Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding](https://arxiv.org/abs/2105.12723)_ by Zhang et al. NesT partitions the input into non-overlapping blocks to perform local self-attention and aggregates them using a simple aggregation layer.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/nest.py).

• ```nest_tiny_224```: NesT-Tiny, resolution 224 x 224.

• ```nest_small_224```: NesT-Small, resolution 224 x 224.

• ```nest_base_224```: NesT-Base, resolution 224 x 224.

## PVT V2

Pyramid vision transformer V2 (PVT V2) from _[PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)_ by Wang et al. PVT V2's primary differences from regular ViTs are spatial reduction attention, overlapping patch embedding, and a convolution in every MLP.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/pvt_v2.py).

• ```pvtv2_b0```: PVTV2-B0.

• ```pvtv2_b1```: PVTV2-B1.

• ```pvtv2_b2```: PVTV2-B2.

• ```pvtv2_b3```: PVTV2-B3.

• ```pvtv2_b4```: PVTV2-B4.

• ```pvtv2_b5```: PVTV2-B5.

## ResNet

ResNet from _[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)_ by He et al. ResNet proposes residual connections
to enable information to propagate more freely throughout the network.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```resnet18```: ResNet-18.

• ```resnet34```: ResNet-34.

• ```resnet26```: ResNet-26.

• ```resnet50```: ResNet-50.

• ```resnet101```: ResNet-101.

• ```resnet152```: ResNet-152.

• ```resnet18_ssl```: ResNet-18, trained using semi-supervised learning, with a teacher model, on a subset of YFCC100M and fine-tuned on ImageNet1K.

• ```resnet50_ssl```: ResNet-50, trained using semi-supervised learning, with a teacher model, on a subset of YFCC100M and fine-tuned on ImageNet1K.

• ```resnet18_swsl```: ResNet-18, trained using semi-weakly supervised learning, with a teacher model, on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnet50_swsl```: ResNet-50, trained using semi-weakly supervised learning, with a teacher model, on 940 million Instagram images and fine-tuned on ImageNet1K.

## ResNet-D

ResNet-D from _[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)_ by He et al. ResNet-D
is identical to ResNet, except that it supplants the 7 x 7 convolution in the stem with three 3 x 3 convolutions
and delegates spatial reduction in the non-identity branch of each residual block to average pooling in lieu of a strided
1 x 1 convolution.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```resnet18d```: ResNet-D-18.

• ```resnet34d```: ResNet-D-34.

• ```resnet26d```: ResNet-D-26.

• ```resnet50d```: ResNet-D-50.

• ```resnet101d```: ResNet-D-101.

• ```resnet152d```: ResNet-D-152.

• ```resnet200d```: ResNet-D-200.

## ResNet-T

ResNet-T from _[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)_ by Wightman. ResNet-T is identical to ResNet-D, except that the
stem is tiered, i.e., the output dimension of the convolutions are 24, 32, and 64 instead of 32, 32, and 64.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```resnet10t```: ResNet-T-10.

• ```resnet14t```: ResNet-T-14.

• ```resnet26t```: ResNet-T-26.

## Wide ResNet

Wide ResNet from _[Wide Residual Networks](https://arxiv.org/abs/1605.07146)_ by Zagoruyko et al. Wide ResNet recommends increasing the number of channels
in the bottleneck dimension of each residual block as an alternative to deepening ResNets.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```wide_resnet50_2```: Wide ResNet-50, width multiplier 2x.

• ```wide_resnet101_2```: Wide ResNet-101, width multiplier 2x.

## ResNeXt

ResNeXt from _[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)_ by Xie et al. ResNeXt is inspired by Inception's split-transform-merge strategy and has a multi-branch design that aggregates a set of homogeneous transformations in each residual block for better performance. 
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```resnext50_32x4d```: ResNeXt-50, cardinality 32, bottleneck dimension of each branch in the first stage 4.

• ```resnext101_32x8d```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 8.

• ```resnext101_64x4d```: ResNeXt-101, cardinality 64, bottleneck dimension of each branch in the first stage 4.

• ```resnext50_32x4d_ssl```: ResNeXt-50, cardinality 32, bottleneck dimension of each branch in the first stage 4, trained using semi-supervised learning, with a teacher model, on a subset of YFCC100M and fine-tuned on ImageNet1K.

• ```resnext101_32x4d_ssl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 4, trained using semi-supervised learning, with a teacher model, on a subset of YFCC100M and fine-tuned on ImageNet1K.

• ```resnext101_32x8d_ssl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 8, trained using semi-supervised learning, with a teacher model, on a subset of YFCC100M and fine-tuned on ImageNet1K.

• ```resnext101_32x16d_ssl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 16, trained using semi-supervised learning, with a teacher model, on a subset of YFCC100M and fine-tuned on ImageNet1K.

• ```resnext50_32x4d_swsl```: ResNeXt-50, cardinality 32, bottleneck dimension of each branch in the first stage 4, trained using semi-weakly supervised learning, with a teacher model, on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnext101_32x4d_swsl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 4, trained using semi-weakly supervised learning, with a teacher model, on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnext101_32x8d_swsl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 8, trained using semi-weakly supervised learning, with a teacher model, on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnext101_32x16d_swsl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 16, trained using semi-weakly supervised learning, with a teacher model, on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnext101_32x8d_wsl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 8, trained using weakly-supervised learning on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnext101_32x16d_wsl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 16, trained using weakly-supervised learning on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnext101_32x32d_wsl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 32, trained using weakly-supervised learning on 940 million Instagram images and fine-tuned on ImageNet1K.

• ```resnext101_32x48d_wsl```: ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 48, trained using weakly-supervised learning on 940 million Instagram images and fine-tuned on ImageNet1K.

## RegNet

RegNet from _[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)_ by Radosavovic et al. RegNet is a family of ResNeXt-like networks whose depths and widths are parameterized through a simple linear function that works well under a variety of FLOPS settings.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/regnet.py).

```regnetx_200mf```: RegNetX costing approximately 200 mega FLOPS/0.2 giga FLOPS. 

```regnetx_400mf```: RegNetX costing approximately 400 mega FLOPS/0.4 giga FLOPS. 

```regnetx_600mf```: RegNetX costing approximately 600 mega FLOPS/0.6 giga FLOPS. 

```regnetx_800mf```: RegNetX costing approximately 800 mega FLOPS/0.8 giga FLOPS. 

```regnetx_1600mf```: RegNetX costing approximately 1600 mega FLOPS/1.6 giga FLOPS. 

```regnetx_3200mf```: RegNetX costing approximately 3200 mega FLOPS/3.2 giga FLOPS. 

```regnetx_4000mf```: RegNetX costing approximately 4000 mega FLOPS/4.0 giga FLOPS. 

```regnetx_6400mf```: RegNetX costing approximately 6400 mega FLOPS/6.4 giga FLOPS. 

```regnetx_8000mf```: RegNetX costing approximately 8000 mega FLOPS/8.0 giga FLOPS. 

```regnetx_12gf```: RegNetX costing approximately 12000 mega FLOPS/12 giga FLOPS. 

```regnetx_16gf```: RegNetX costing approximately 16000 mega FLOPS/16 giga FLOPS.

```regnetx_32gf```: RegNetX costing approximately 32000 mega FLOPS/32 giga FLOPS.

```regnety_200mf```: RegNetY costing approximately 200 mega FLOPS/0.2 giga FLOPS. 

```regnety_400mf```: RegNetY costing approximately 400 mega FLOPS/0.4 giga FLOPS. 

```regnety_600mf```: RegNetY costing approximately 600 mega FLOPS/0.6 giga FLOPS. 

```regnety_800mf```: RegNetY costing approximately 800 mega FLOPS/0.8 giga FLOPS. 

```regnety_1600mf```: RegNetY costing approximately 1600 mega FLOPS/1.6 giga FLOPS. 

```regnety_3200mf```: RegNetY costing approximately 3200 mega FLOPS/3.2 giga FLOPS. 

```regnety_4000mf```: RegNetY costing approximately 4000 mega FLOPS/4.0 giga FLOPS. 

```regnety_6400mf```: RegNetY costing approximately 6400 mega FLOPS/6.4 giga FLOPS. 

```regnety_8000mf```: RegNetY costing approximately 8000 mega FLOPS/8.0 giga FLOPS. 

```regnety_12gf```: RegNetY costing approximately 12000 mega FLOPS/12 giga FLOPS. 

```regnety_16gf```: RegNetY costing approximately 16000 mega FLOPS/16 giga FLOPS. 

```regnety_32gf```: RegNetY costing approximately 32000 mega FLOPS/32 giga FLOPS. 

## SENet

Squeeze-and-excitation network (SENet) from _[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)_ by Hu et al. SENet introduces squeeze-and-excitation (SE), a channel attention mechanism, for capturing relationships between different channels and adaptively weighing them. 
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```seresnet50```: SE-ResNet-50.

• ```seresnet152d```: SE-ResNet-D-152.

• ```seresnext50_32x4d```: SE-ResNeXt-50, cardinality 32, bottleneck dimension of each branch in the first stage 4.

• ```seresnext101_32x8d```: SE-ResNeXt-101, cardinality 32, bottleneck dimension of each branch in the first stage 8.

• ```seresnext26d_32x4d```: SE-ResNeXt-D-26, cardinality 32, bottleneck dimension of each branch in the first stage 4.

• ```seresnext101d_32x8d```: SE-ResNeXt-D-101, cardinality 32, bottleneck dimension of each branch in the first stage 8.

• ```seresnext26t_32x4d```: SE-ResNeXt-T-26, cardinality 32, bottleneck dimension of each branch in the first stage 4.

## ECANet

Efficient channel attention network (ECANet) from _[ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)_ by Wang et al. ECANet suggests an alternative to squeeze-and-excitation, dubbed efficient channel attention (ECA), that eliminates dimensionality reduction in the bottleneck layer of the excitation module
for better accuracy, in addition to being cheaper.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```ecaresnet50_light```: Lightweight ECA-ResNet-50 with most of the layers being in stage 3.

• ```ecaresnet50d```: ECA-ResNet-D-50.

• ```ecaresnet101d```: ECA-ResNet-D-101.

• ```ecaresnet269d```: ECA-ResNet-D-269.

• ```ecaresnet26t```: ECA-ResNet-T-26.

• ```ecaresnet50t```: ECA-ResNet-T-50.

## ResNet-RS

ResNet-RS from _[Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)_ by Bello et al. ResNet-RS is similar  to an 
SE-ResNet-D but enjoys a refined scaling procedure.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

• ```resnetrs50```: ResNet-RS-50.

• ```resnetrs101```: ResNet-RS-101.

• ```resnetrs152```: ResNet-RS-152.

• ```resnetrs200```: ResNet-RS-200.

• ```resnetrs270```: ResNet-RS-270.

• ```resnetrs350```: ResNet-RS-350.

• ```resnetrs420```: ResNet-RS-420.

## SKNet

Selective kernel networks (SKNet) from _[Selective Kernel Networks](https://arxiv.org/abs/1903.06586)_ by Li et al. SKNet uses selective kernel units (SK units) to adjust each residual block's
kernel size according to the input.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/sknet.py).

• ```skresnet18```: SK-ResNet-18.

• ```skresnet34```: SK-Resnet-34.

• ```skresnext50_32x4d```: SK-ResNeXt-50, cardinality 32, bottleneck dimension of each branch in the first stage 4.

## ResNeSt

ResNeSt from _[ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)_ by Zhang et al. ResNeSt compounds the power of multi-branch architectures with that of channel attention
using a mechanism known as split attention (SplAt).
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnest.py).

• ```resnest14_2s1x64d```: ResNeSt-14, radix 2, cardinality 1, bottleneck dimension of each branch in the first stage 64.

• ```resnest26_2s1x64d```: ResNeSt-26, radix 2, cardinality 1, bottleneck dimension of each branch in the first stage 64.

• ```resnest50_2s1x64d```: ResNeSt-50, radix 2, cardinality 1, bottleneck dimension of each branch in the first stage 64.

• ```resnest101_2s1x64d```: ResNeSt-101, radix 2, cardinality 1, bottleneck dimension of each branch in the first stage 64.

• ```resnest200_2s1x64d```: ResNeSt-200, radix 2, cardinality 1, bottleneck dimension of each branch in the first stage 64.

• ```resnest269_2s1x64d```: ResNeSt-269, radix 2, cardinality 1, bottleneck dimension of each branch in the first stage 64.

## Swin

Shifted window attention transformer (Swin transformer) from _[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)_ by Liu et al. Swin is a hierarchical vision transformer that uses
shifted window attention (Swin attention) to parsimoniously calculate attention by limiting it to local windows whilst also capturing cross-window relationships.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/swin_transformer.py).

• ```swin_tiny_window7_224```: Swin-Tiny, window size 7 x 7, resolution 224 x 224.

• ```swin_small_window7_224```: Swin-Small, window size 7 x 7, resolution 224 x 224.

• ```swin_base_window7_224```: Swin-Base, window size 7 x 7, resolution 224 x 224.

• ```swin_large_window7_224```: Swin-Large, window size 7 x 7, resolution 224 x 224.

• ```swin_base_window12_384```: Swin-Base, window size 12 x 12, resolution 384 x 384.

• ```swin_large_window12_384```: Swin-Large, window size 12 x 12, resolution 384 x 384.

• ```swin_base_window7_224_in22k```: Swin-Base, window size 7 x 7, resolution 224 x 224, trained on ImageNet22K.

• ```swin_large_window7_224_in22k```: Swin-Large, window size 7 x 7, resolution 224 x 224, trained on ImageNet22K.

• ```swin_base_window12_384_in22k```: Swin-Base, window size 7 x 7, resolution 384 x 384, trained on ImageNet22K.

• ```swin_large_window12_384_in22k```: Swin-Large, window size 7 x 7, resolution 384 x 384, trained on ImageNet22K.

## Swin-S3

Swin-S3 from _[Searching the Search Space of Vision Transformer](https://arxiv.org/abs/2111.14725)_ by Chen et al. Swin-S3 is composed of the same building blocks as Swin but was discovered through neural architecture search.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/swin_transformer.py).

• ```swin_s3_tiny_224```: Swin-S3-Tiny, resolution 224 x 224. 

• ```swin_s3_small_224```: Swin-S3-Small, resolution 224 x 224. 

• ```swin_s3_base_224```: Swin-S3-Base, resolution 224 x 224. 

## VAN

Visual attention network (VAN) from _[Visual Attention Network](https://arxiv.org/abs/2202.09741)_ by Guo et al. VAN combines the merits of convolutions and self-attention through large kernel attention (LKA), a linear attention module with spatial and channel adaptability as well as the ability to capture long-range dependencies.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/Visual-Attention-Network/VAN-Classification).

• ```van_b0```: VAN-B0.

• ```van_b1```: VAN-B1.

• ```van_b2```: VAN-B2.

• ```van_b3```: VAN-B3.

## VGG

VGG from _[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)_ by Simonyan et al. VGG stacks many 3 x 3 convolutions on top of one another and showed the importance of depth for object recognition.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vgg.py).

• ```vgg11```: VGG-11.

• ```vgg13```: VGG-13.

• ```vgg16```: VGG-16.

• ```vgg19```: VGG-19.

• ```vgg11_bn```: VGG-11 with batch normalization.

• ```vgg13_bn```: VGG-13 with batch normalization.

• ```vgg16_bn```: VGG-16 with batch normalization.

• ```vgg19_bn```: VGG-19 with batch normalization.

## ViT

Vision transformer (ViT) from _[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)_ by Dosovitskiy et al. ViT is identical to an NLP transformer, except that the input is first patchified and linearly transformed to manage the quadratic complexity of self-attention. 
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py).

• ```vit_small_patch32_224```: ViT-Small, patch size 32 x 32, resolution 224 x 224.

• ```vit_base_patch32_224```: ViT-Base, patch size 32 x 32, resolution 224 x 224.

• ```vit_tiny_patch16_224```: ViT-Tiny, patch size 16 x 16, resolution 224 x 224.

• ```vit_small_patch16_224```: ViT-Small, patch size 16 x 16, resolution 224 x 224.

• ```vit_base_patch16_224```: ViT-Base, patch size 16 x 16, resolution 224 x 224.

• ```vit_large_patch16_224```: ViT-Large, patch size 16 x 16, resolution 224 x 224.

• ```vit_base_patch8_224```: ViT-Base, patch size 8 x 8, resolution 224 x 224.

• ```vit_small_patch32_384```: ViT-Small, patch size 32 x 32, resolution 384 x 384.

• ```vit_base_patch32_384```: ViT-Base, patch size 32 x 32, resolution 384 x 384.

• ```vit_large_patch32_384```: ViT-Large, patch size 32 x 32, resolution 384 x 384.

• ```vit_tiny_patch16_384```: ViT-Tiny, patch size 16 x 16, resolution 384 x 384.

• ```vit_small_patch16_384```: ViT-Small, patch size 16 x 16, resolution 384 x 384.

• ```vit_base_patch16_384```: ViT-Base, patch size 16 x 16, resolution 384 x 384.

• ```vit_large_patch16_384```: ViT-Large, patch size 16 x 16, resolution 384 x 384.

• ```vit_small_patch32_224_in22k```: ViT-Small, patch size 32 x 32, resolution 224 x 224, trained on ImageNet22K.

• ```vit_base_patch32_224_in22k```: ViT-Base, patch size 32 x 32, resolution 224 x 224, trained on ImageNet22K.

• ```vit_large_patch32_224_in22k```: ViT-Large, patch size 32 x 32, resolution 224 x 224, trained on ImageNet22K.

• ```vit_tiny_patch16_224_in22k```: ViT-Tiny, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

• ```vit_small_patch16_224_in22k```: ViT-Small, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

• ```vit_base_patch16_224_in22k```: ViT-Base, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

• ```vit_large_patch16_224_in22k```: ViT-Large, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

• ```vit_huge_patch14_224_in22k```: ViT-Huge, patch size 14 x 14, resolution 224 x 224, trained on ImageNet22K.

• ```vit_base_patch8_224_in22k```: ViT-Base, patch size 8 x 8, resolution 224 x 224, trained on ImageNet22K.

## ViT SAM

ViTs trained using sharpness-aware minimization (SAM) from _[When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)_ by Chen et al. SAM can be used to train ViTs that outperform ResNets without the need for large-scale pre-training or strong data augmentation.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py).

• ```vit_base_patch32_224_sam```: ViT-Base, patch size 32 x 32, resolution 224 x 224.

• ```vit_base_patch16_224_sam```: ViT-Base, patch size 16 x 16, resolution 224 x 224.

## ViT DINO

ViTs trained using self-distillation with no labels (DINO) from _[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)_ by Caron et al. DINO is a self-supervised learning technique that is particularly suited to vision transformers and exhibits interesting properties such as the emergence of unsupervised segmentation.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py).

• ```vit_small_patch16_224_dino```: ViT-Small, patch size 16 x 16, resolution 224 x 224.

• ```vit_base_patch16_224_dino```: ViT-Base, patch size 16 x 16, resolution 224 x 224.

• ```vit_small_patch8_224_dino```: ViT-Small, patch size 8 x 8, resolution 224 x 224.

• ```vit_base_patch8_224_dino```: ViT-Base, patch size 8 x 8, resolution 224 x 224.

## DeiT 3

Data-efficient image transformer III (DeiT 3) from _[DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118)_ by Touvron et al. DeiT 3 is very similar to ViT but is trained using a better recipe.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/deit.py).

• ```deit3_small_patch16_224```: DeiT3-Small, patch size 16 x 16, resolution 224 x 224.

• ```deit3_medium_patch16_224```: DeiT3-Medium, patch size 16 x 16, resolution 224 x 224.

• ```deit3_base_patch16_224```: DeiT3-Base, patch size 16 x 16, resolution 224 x 224.

• ```deit3_large_patch16_224```: DeiT3-Large, patch size 16 x 16, resolution 224 x 224.

• ```deit3_huge_patch14_224```: DeiT3-Huge, patch size 14 x 14, resolution 224 x 224.

• ```deit3_small_patch16_384```: DeiT3-Small, patch size 16 x 16, resolution 384 x 384.

• ```deit3_base_patch16_384```: DeiT3-Base, patch size 16 x 16, resolution 384 x 384.

• ```deit3_large_patch16_384```: DeiT3-Large, patch size 16 x 16, resolution 384 x 384.

• ```deit3_small_patch16_224_in22ft1k```: DeiT3-Small, patch size 16 x 16, resolution 224 x 224, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

• ```deit3_medium_patch16_224_in22ft1k```: DeiT3-Medium, patch size 16 x 16, resolution 224 x 224, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

• ```deit3_base_patch16_224_in22ft1k```: DeiT3-Base, patch size 16 x 16, resolution 224 x 224, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

• ```deit3_large_patch16_224_in22ft1k```: DeiT3-Large, patch size 16 x 16, resolution 224 x 224, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

• ```deit3_huge_patch14_224_in22ft1k```: DeiT3-Huge, patch size 14 x 14, resolution 224 x 224, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

• ```deit3_small_patch16_384_in22ft1k```: DeiT3-Small, patch size 16 x 16, resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

• ```deit3_base_patch16_384_in22ft1k```: DeiT3-Base, patch size 16 x 16, resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

• ```deit3_large_patch16_384_in22ft1k```: DeiT3-Large, patch size 16 x 16, resolution 384 x 384, pre-trained on ImageNet22K and fine-tuned on Imagenet1K.

## BEiT

BEiT from _[BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)_ by Bao et al. BEiT is very similar to ViT with relative position embedding but is trained using BERT-style masked modelling.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/beit.py).

• ```beit_base_patch16_224```: BEiT-Base, patch size 16 x 16, resolution 224 x 224. 

• ```beit_large_patch16_224```: BEiT-Large, patch size 16 x 16, resolution 224 x 224. 

• ```beit_base_patch16_384```: BEiT-Base, patch size 16 x 16, resolution 384 x 384. 

• ```beit_large_patch16_384```: BEiT-Large, patch size 16 x 16, resolution 384 x 384. 

• ```beit_large_patch16_512```: BEiT-Large, patch size 16 x 16, resolution 512 x 512. 

• ```beit_base_patch16_224_in22k```: BEiT-Base, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

• ```beit_large_patch16_224_in22k```: BEiT-Large, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

## BEiT V2

BEiT V2 from _[BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)_ by Peng et al. BEiT V2 is identical to BEiT in terms of architecture but uses a different masked modelling approach that relies on a vector-quantized knowledge distillation algorithm for exploiting higher-level semantics.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/beit.py).

• ```beitv2_base_patch16_224```: BEiTV2-Base, patch size 16 x 16, resolution 224 x 224.

• ```beitv2_large_patch16_224```: BEiTV2-Large, patch size 16 x 16, resolution 224 x 224.

• ```beitv2_base_patch16_224_in22k```: BEiTV2-Base, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

• ```beitv2_large_patch16_224_in22k```: BEiTV2-Large, patch size 16 x 16, resolution 224 x 224, trained on ImageNet22K.

## XCiT

Cross-covariance image transformer (XCiT) from _[XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)_ by El-Nouby et al. XCiT
introduces cross-covariance attention (XCA), an inverted version of self-attention that is applied along the feature axis rather than the token axis to eliminate the quadratix complexity of traditional transformers, and complements it with local patch interaction modules (LPI) that explicitly enable communications amongst patches for scalable, efficient image transformers.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/xcit.py).<br>

```xcit_nano12_patch16_224```: XCiT-Nano, depth 12, patch size 16 x 16, training resolution 224 x 224.

```xcit_nano12_patch16_224_dist```: XCiT-Nano, depth 12, patch size 16 x 16, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_nano12_patch16_384_dist```: XCiT-Nano, depth 12, patch size 16 x 16, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_nano12_patch8_224```: XCiT-Nano, depth 12, patch size 8 x 8, training resolution 224 x 224.

```xcit_nano12_patch8_224_dist```: XCiT-Nano, depth 12, patch size 8 x 8, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_nano12_patch8_384_dist```: XCiT-Nano, depth 12, patch size 8 x 8, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_tiny12_patch16_224```: XCiT-Tiny, depth 12, patch size 16 x 16, training resolution 224 x 224.

```xcit_tiny12_patch16_224_dist```: XCiT-Tiny, depth 12, patch size 16 x 16, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_tiny12_patch16_384_dist```: XCiT-Tiny, depth 12, patch size 16 x 16, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_tiny24_patch16_224```: XCiT-Tiny, depth 24, patch size 16 x 16, training resolution 224 x 224.

```xcit_tiny24_patch16_224_dist```: XCiT-Tiny, depth 24, patch size 16 x 16, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_tiny24_patch16_384_dist```: XCiT-Tiny, depth 24, patch size 16 x 16, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_tiny12_patch8_224```: XCiT-Tiny, depth 12, patch size 8 x 8, training resolution 224 x 224.

```xcit_tiny12_patch8_224_dist```: XCiT-Tiny, depth 12, patch size 8 x 8, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_tiny12_patch8_384_dist```: XCiT-Tiny, depth 12, patch size 8 x 8, training resolution 384 x 384, trained using distillation with a convolutional teacher.

```xcit_tiny24_patch8_224```: XCiT-Tiny, depth 24, patch size 8 x 8, training resolution 224 x 224.

```xcit_tiny24_patch8_224_dist```: XCiT-Tiny, depth 24, patch size 8 x 8, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_tiny24_patch8_384_dist```: XCiT-Tiny, depth 24, patch size 8 x 8, training resolution 384 x 384, trained using distillation with a convolutional teacher.

```xcit_small12_patch16_224```: XCiT-Small, depth 12, patch size 16 x 16, training resolution 224 x 224.

```xcit_small12_patch16_224_dist```: XCiT-Small, depth 12, patch size 16 x 16, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_small12_patch16_384_dist```: XCiT-Small, depth 12, patch size 16 x 16, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_small24_patch16_224```: XCiT-Small, depth 24, patch size 16 x 16, training resolution 224 x 224.

```xcit_small24_patch16_224_dist```: XCiT-Small, depth 24, patch size 16 x 16, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_small24_patch16_384_dist```: XCiT-Small, depth 24, patch size 16 x 16, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_small12_patch8_224```: XCiT-Small, depth 12, patch size 8 x 8, training resolution 224 x 224.

```xcit_small12_patch8_224_dist```: XCiT-Small, depth 12, patch size 8 x 8, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_small12_patch8_384_dist```: XCiT-Small, depth 12, patch size 8 x 8, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_small24_patch8_224```: XCiT-Small, depth 24, patch size 8 x 8, training resolution 224 x 224.

```xcit_small24_patch8_224_dist```: XCiT-Small, depth 24, patch size 8 x 8, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_small24_patch8_384_dist```: XCiT-Small, depth 24, patch size 8 x 8, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_medium24_patch16_224```: XCiT-Medium, depth 24, patch size 16 x 16, training resolution 224 x 224.

```xcit_medium24_patch16_224_dist```: XCiT-Medium, depth 24, patch size 16 x 16, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_medium24_patch16_384_dist```: XCiT-Medium, depth 24, patch size 16 x 16, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_medium24_patch8_224```: XCiT-Medium, depth 24, patch size 8 x 8, training resolution 224 x 224.

```xcit_medium24_patch8_224_dist```: XCiT-Medium, depth 24, patch size 8 x 8, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_medium24_patch8_384_dist```: XCiT-Medium, depth 24, patch size 8 x 8, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_large24_patch16_224```: XCiT-Large, depth 24, patch size 16 x 16, training resolution 224 x 224.

```xcit_large24_patch16_224_dist```: XCiT-Large, depth 24, patch size 16 x 16, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_large24_patch16_384_dist```: XCiT-Large, depth 24, patch size 16 x 16, training resolution 384 x 384, trained using distillation with a convolutional teacher. 

```xcit_large24_patch8_224```: XCiT-Large, depth 24, patch size 8 x 8, training resolution 224 x 224.

```xcit_large24_patch8_224_dist```: XCiT-Large, depth 24, patch size 8 x 8, training resolution 224 x 224, trained using distillation with a convolutional teacher.

```xcit_large24_patch8_384_dist```: XCiT-Large, depth 24, patch size 8 x 8, training resolution 384 x 384, trained using distillation with a convolutional teacher. 
