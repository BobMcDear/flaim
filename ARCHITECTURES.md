# Available Architectures

The following is a list of all available architectures and their pre-trained parameters, with brief descriptions and references, classified into network families. Unless otherwise mentioned, the pre-trained parameters were learned through supervised training.

&#x25cf; <strong>[CaiT](#cait)</strong><br>
&#x25cf; <strong>[ConvMixer](#convmixer)</strong><br>
&#x25cf; <strong>[ConvNeXt](#convnext)</strong><br>
&#x25cf; <strong>[ConvNeXt V2](#convnext-v2)</strong><br>
&#x25cf; <strong>[DaViT](#davit)</strong><br>
&#x25cf; <strong>[EfficientNetV2](#efficientnetv2)</strong><br>
&#x25cf; <strong>[GC ViT](#gc-vit)</strong><br>
&#x25cf; <strong>[HorNet](#hornet)</strong><br>
&#x25cf; <strong>[MaxViT](#maxvit)</strong><br>
&#x25cf; <strong>[NesT](#nest)</strong><br>
&#x25cf; <strong>[PiT](#pit)</strong><br>
&#x25cf; <strong>[PVT V2](#pvt-v2)</strong><br>
&#x25cf; <strong>[RegNet](#regnet)</strong><br>
&#x25cf; <strong>[ResNet](#resnet)</strong><br>
&#x25cf; <strong>[ResNet-D](#resnet-d)</strong><br>
&#x25cf; <strong>[ResNet-T](#resnet-t)</strong><br>
&#x25cf; <strong>[Wide ResNet](#wide-resnet)</strong><br>
&#x25cf; <strong>[ResNeXt](#resnext)</strong><br>
&#x25cf; <strong>[SENet](#senet)</strong><br>
&#x25cf; <strong>[ECANet](#ecanet)</strong><br>
&#x25cf; <strong>[ResNet-RS](#resnet-rs)</strong><br>
&#x25cf; <strong>[SKNet](#sknet)</strong><br>
&#x25cf; <strong>[ResNeSt](#resnest)</strong><br>
&#x25cf; <strong>[Swin](#swin)</strong><br>
&#x25cf; <strong>[Swin-S3](#swin-s3)</strong><br>
&#x25cf; <strong>[VAN](#van)</strong><br>
&#x25cf; <strong>[VGG](#vgg)</strong><br>
&#x25cf; <strong>[ViT](#vit)</strong><br>
&#x25cf; <strong>[DeiT 3](#deit-3)</strong><br>
&#x25cf; <strong>[BEiT](#beit)</strong><br>
&#x25cf; <strong>[XCiT](#xcit)</strong><br>

## CaiT

Class attention image transformer (CaiT) from _[Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)_ by Touvron et al. CaiT
presents two novel modules, LayerScale and class attention, that enable ViTs to go significantly deeper with little saturation in accuracy
at greater depths.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/cait.py).

&#x25cf; ```cait_xxsmall24```: CaiT-XXSmall, depth 24.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.

&#x25cf; ```cait_xxsmall36```: CaiT-XXSmall, depth 36.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.

&#x25cf; ```cait_small24_224```: CaiT-Small, depth 24.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.

&#x25cf; ```cait_xsmall24```: CaiT-XSmall, depth 24.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.

&#x25cf; ```cait_small36```: CaiT-Small, depth 36.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.

&#x25cf; ```cait_medium36```: CaiT-Medium, depth 36.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.

&#x25cf; ```cait_medium48```: CaiT-Medium, depth 48.<br>
* ```in1k_448```: Trained on ImageNet1K at resolution 448 x 448.

## ConvMixer

ConvMixer from _[Patches Are All You Need?](https://arxiv.org/abs/2201.09792)_ by Trockman et al. ConvMixer
is similar to isotropic architectures like ViT but uses convolutions with large kernel sizes to perform token mixing.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convmixer.py).

&#x25cf; ```convmixer20_1024d_patch14_kernel9```: ConvMixer, depth 20, token dimension 1024, patch size 14 x 14, kernel size 9 x 9.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convmixer20_1536d_patch7_kernel9```: ConvMixer, depth 20, token dimension 1536, patch size 7 x 7, kernel size 9 x 9.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convmixer32_768d_patch7_kernel7```: ConvMixer, depth 32, token dimension 768, patch size 7 x 7, kernel size 7 x 7.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ConvNeXt

ConvNeXt from _[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)_ by Liu et al. ConvNeXt borrows ideas from the vision transformer literature, such as larger kernel sizes and more aggressive downsampling in the stem, to modernize a plain ResNet and attain results on par with state-of-the-art
vision transformers like Swin using a purely convolutional network.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convnext.py).

&#x25cf; ```convnext_atto```: ConvNeXt-Atto.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convnext_femto```: ConvNeXt-Femto.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convnext_pico```: ConvNeXt-Pico.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convnext_nano```: ConvNeXt-Nano.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in12k_224```: Trained on ImageNet12K at resolution 224 x 224.<br>
* ```in12k_ft_in1k_224```: Pre-trained on ImageNet12K and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convnext_tiny```: ConvNeXt-Tiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in12k_224```: Trained on ImageNet12K at resolution 224 x 224.<br>
* ```in12k_ft_in1k_224```: Pre-trained on ImageNet12K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in12k_ft_in1k_384```: Pre-trained on ImageNet12K and fine-tuned on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnext_small```: ConvNeXt-Small.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in12k_224```: Trained on ImageNet12K at resolution 224 x 224.<br>
* ```in12k_ft_in1k_224```: Pre-trained on ImageNet12K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in12k_ft_in1k_384```: Pre-trained on ImageNet12K and fine-tuned on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnext_base```: ConvNeXt-Base.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.<br>
* ```clip_laion2b_256```: Trained using CLIP on LAION-2B at resolution 256 x 256.<br>
* ```clip_laion2b_augreg_256```: Trained using CLIP on LAION-2B with additional augmentation & regularization at resolution 256 x 256.<br>
* ```clip_laiona_256```: Trained using CLIP on LAION-Aesthetics at resolution 256 x 256.<br>
* ```clip_laiona_320```: Trained using CLIP on LAION-Aesthetics at resolution 320 x 320.<br>
* ```clip_laiona_augreg_320```: Trained using CLIP on LAION-Aesthetics with additional augmentation & regularization at resolution 320 x 320.<br>
* ```clip_laion2b_augreg_ft_in1k_256```: Pre-trained using CLIP on LAION-2B with additional augmentation & regularization and fine-tuned on ImageNet1K at resolution 256 x 256.<br>
* ```clip_laiona_augreg_ft_in1k_384```: Pre-trained using CLIP on LAION-Aesthetics with additional augmentation & regularization and fine-tuned on ImageNet1K at resolution 384 x 384.<br>

&#x25cf; ```convnext_large```: ConvNeXt-Large.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnext_large_mlp```: ConvNeXt-Large with an MLP in the head.<br>
* ```clip_laion2b_augreg_256```: Trained using CLIP on LAION-2B with additional augmentation & regularization at resolution 256 x 256.<br>
* ```clip_laion2b_augreg_256_ft_320```: Trained using CLIP on LAION-2B with additional augmentation & regularization at resolution 256 x 256 and fine-tuned at resolution 320 x 320.<br>
* ```clip_laion2b_soup_augreg_256_ft_320```: A soup (i.e., parameters averaged) of 3 models trained using CLIP on LAION-2B with additional augmentation & regularization at resolution 256 x 256 and fine-tuned at resolution 320 x 320.<br>
* ```clip_laion2b_augreg_ft_in1k_256```: Pre-trained using CLIP on LAION-2B with additional augmentation & regularization and fine-tuned on ImageNet1K at resolution 256 x 256.<br>
* ```clip_laion2b_augreg_ft_in1k_384```: Pre-trained using CLIP on LAION-2B with additional augmentation & regularization and fine-tuned on ImageNet1K at resolution 384 x 384.<br>

&#x25cf; ```convnext_xlarge```: ConvNeXt-XLarge.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnext_xxlarge```: ConvNeXt-XXLarge.<br>
* ```clip_laion2b_rewind_256```: Trained using CLIP on LAION-2B at resolution 256 x 256, last 10% of training rewinded and resumed with slightly different settings.<br>
* ```clip_laion2b_soup_256```: Trained using CLIP on LAION-2B at resolution 256 x 256, a soup (i.e., parameters averaged) of the original and rewinded training runs.<br>

## ConvNeXt V2

ConvNeXt V2 from _[ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)_ by Woo et al. ConvNeXt V2 is based on ConvNeXt but is trained using a fully convolutional masked autoencoder training scheme (FCMAE) and also incorporates global response normalization (GRN), a normalization module that abates inter-channel feature redundancies and is particularly important for effective learning with FCMAE.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/convnext.py).

&#x25cf; ```convnextv2_atto```: ConvNeXtV2-Atto.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convnextv2_femto```: ConvNeXtV2-Femto.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convnextv2_pico```: ConvNeXtV2-Pico.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.

&#x25cf; ```convnextv2_nano```: ConvNeXtV2-Nano.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_384```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnextv2_tiny```: ConvNeXtV2-Tiny.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_384```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnextv2_base```: ConvNeXtV2-Base.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_384```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnextv2_large```: ConvNeXtV2-Large.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_384```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 384 x 384.

&#x25cf; ```convnextv2_huge```: ConvNeXtV2-Huge.<br>
* ```fcmae_in1k_224```: Trained using FCMAE with no labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in1k_ft_in1k_224```: Pre-trained using FCMAE with no labels on ImageNet1K and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_384```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 384 x 384.<br>
* ```fcmae_in22k_ft_in22k_ft_in1k_512```: Pre-trained using FCMAE with no labels on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 512 x 512.

## DaViT

DaViT from _[DaViT: Dual Attention Vision Transformers](https://arxiv.org/abs/2204.03645)_ by Ding et al. DaViT captures local spatial interactions through window attention and avails of channel self-attention to model global interactions whilst maintaining a linear complexity.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/davit.py).

&#x25cf; ```davit_tiny```: DaViT-Tiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```davit_small```: DaViT-Small.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```davit_base```: DaViT-Base.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## EfficientNetV2

EfficientNetV2 from _[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)_ by Tan et al. EfficientNetV2
builds on EfficientNet but refines the architecture search space by introducing fused MBConv to it, takes training speed into account, and more.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/efficientnet.py).

&#x25cf; ```efficientnetv2_small```: EfficientNetV2-Small.<br>
* ```in1k_300```: Trained on ImageNet1K at resolution 300 x 300.<br>
* ```in22k_300```: Trained on ImageNet22K at resolution 300 x 300.<br>
* ```in22k_ft_in1k_300```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 300 x 300.

&#x25cf; ```efficientnetv2_medium```: EfficientNetV2-Medium.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_384```: Trained on ImageNet22K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```efficientnetv2_large```: EfficientNetV2-Large.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_384```: Trained on ImageNet22K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```efficientnetv2_xlarge```: EfficientNetV2-XLarge.<br>
* ```in22k_384```: Trained on ImageNet22K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

## GC ViT

Global context vision transformer (GC ViT) from _[Global Context Vision Transformers](https://arxiv.org/abs/2206.09959)_ by Hatamizadeh et al. GC ViT's core contribution is global context attention, where global queries are matched against local keys and values to calculate global spatial relationships, which works in tandem with window attention to efficiently model both long- and short-range interactions without the need for sophisticated tricks like window shifting.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/gcvit.py).

&#x25cf; ```gcvit_xxtiny```: GCViT-XXTiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```gcvit_xtiny```: GCViT-XTiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```gcvit_tiny```: GCViT-Tiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```gcvit_small```: GCViT-Small.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```gcvit_base```: GCViT-Base.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## HorNet

High-order spatial interaction network (HorNet) from _[HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/abs/2207.14284)_ by Rao et al. HorNet uses recursive gated convolutions (g^n convolutions) to model long-range spatial interactions in the input via convolutions.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/raoyongming/HorNet).

&#x25cf; ```hornet_tiny```: HorNet-Tiny, kernel size 7 x 7.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```hornet_small```: HorNet-Small, kernel size 7 x 7.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```hornet_base```: HorNet-Base, kernel size 7 x 7.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```hornet_large```: HorNet-Large, kernel size 7 x 7.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.

## MaxViT

Multi-axis vision transformer (MaxViT) from _[MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697)_ by Tu et al. MaxViT combines
dilated global attention, a method for efficiently expressing global interactions, with MBConv and window attention for an architecture that is able to model global and local spatial relationships and can be scaled to high-resolution input sizes.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/maxxvit.py).

&#x25cf; ```maxvit_tiny```: MaxViT-Tiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in1k_512```: Trained on ImageNet1K at resolution 512 x 512.

&#x25cf; ```maxvit_small```: MaxViT-Small.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in1k_512```: Trained on ImageNet1K at resolution 512 x 512.

&#x25cf; ```maxvit_base```: MaxViT-Base.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in1k_512```: Trained on ImageNet1K at resolution 512 x 512.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_512```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 512 x 512.

&#x25cf; ```maxvit_large```: MaxViT-Large.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in1k_512```: Trained on ImageNet1K at resolution 512 x 512.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_512```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 512 x 512.

&#x25cf; ```maxvit_xlarge```: MaxViT-XLarge.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_512```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 512 x 512.

## NesT

Nested transformer (NesT) from _[Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding](https://arxiv.org/abs/2105.12723)_ by Zhang et al. NesT partitions the input into non-overlapping blocks, separately processes them using transformer layers, and aggregates them via convolutions & pooling to obtain competitive performance - especially on small-scale datasets - without sacrificing the simplicity of the original ViT.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/nest.py).

&#x25cf; ```nest_tiny```: NesT-Tiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```nest_small```: NesT-Small.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```nest_base```: NesT-Base.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## PiT

Pooling-based vision transformer (PiT) from _[Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302)_ by Heo et al. PiT notes that convolutional neural networks, unlike transformers, have a pyramidal configuration where the number of channels gradually increase in later layers whereas the spatial dimensions decrease, and applies a similar design principle to ViTs for better performance.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/pit.py).

&#x25cf; ```pit_tiny```: PiT-Tiny.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pit_xsmall```: PiT-XSmall.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pit_small```: PiT-Small.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pit_base```: PiT-Base.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## PVT V2

Pyramid vision transformer V2 (PVT V2) from _[PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)_ by Wang et al. PVT V2 modifies the self-attention operation by downsampling the input prior to generating keys and values, thereby diminishing the cost of this module, and couples it with convolutions in each MLP for adaptive position encoding and overlapping patch embedding for cutting-edge image recognition.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/pvt_v2.py).

&#x25cf; ```pvtv2_b0```: PVTV2-B0.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pvtv2_b1```: PVTV2-B1.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pvtv2_b2```: PVTV2-B2.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pvtv2_b3```: PVTV2-B3.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pvtv2_b4```: PVTV2-B4.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```pvtv2_b5```: PVTV2-B5.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## RegNet

RegNet from _[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)_ by Radosavovic et al. RegNet is a family of ResNeXt-like networks whose depths and widths are parameterized through a simple linear function that yields a surprisingly powerful collection of networks that work well under a variety of FLOPS settings.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/regnet.py).

&#x25cf; ```regnetx_200mf```: RegNetX costing approximately 200 mega FLOPS/0.2 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_400mf```: RegNetX costing approximately 400 mega FLOPS/0.4 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_600mf```: RegNetX costing approximately 600 mega FLOPS/0.6 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_800mf```: RegNetX costing approximately 800 mega FLOPS/0.8 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_1600mf```: RegNetX costing approximately 1600 mega FLOPS/1.6 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_3200mf```: RegNetX costing approximately 3200 mega FLOPS/3.2 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_4000mf```: RegNetX costing approximately 4000 mega FLOPS/4.0 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_6400mf```: RegNetX costing approximately 6400 mega FLOPS/6.4 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_8000mf```: RegNetX costing approximately 8000 mega FLOPS/8.0 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_12gf```: RegNetX costing approximately 12000 mega FLOPS/12 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_16gf```: RegNetX costing approximately 16000 mega FLOPS/16 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnetx_32gf```: RegNetX costing approximately 32000 mega FLOPS/32 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_200mf```: RegNetY costing approximately 200 mega FLOPS/0.2 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_400mf```: RegNetY costing approximately 400 mega FLOPS/0.4 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_600mf```: RegNetY costing approximately 600 mega FLOPS/0.6 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_800mf```: RegNetY costing approximately 800 mega FLOPS/0.8 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_1600mf```: RegNetY costing approximately 1600 mega FLOPS/1.6 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_3200mf```: RegNetY costing approximately 3200 mega FLOPS/3.2 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_4000mf```: RegNetY costing approximately 4000 mega FLOPS/4.0 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_6400mf```: RegNetY costing approximately 6400 mega FLOPS/6.4 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_8000mf```: RegNetY costing approximately 8000 mega FLOPS/8.0 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_12gf```: RegNetY costing approximately 12000 mega FLOPS/12 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_16gf```: RegNetY costing approximately 16000 mega FLOPS/16 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```regnety_32gf```: RegNetY costing approximately 32000 mega FLOPS/32 giga FLOPS.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ResNet

ResNet from _[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)_ by He et al. ResNet proposes residual connections
to facilitate the propagation of information throughout the network and for the first time rendered possible models of unprecedented depths.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```resnet18```: ResNet-18.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```ssl_ft_in1k_224```: Pre-trained on a subset of YFCC100M using semi-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```swsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using semi-weakly supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet34```: ResNet-34.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet26```: ResNet-26.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet50```: ResNet-50.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```ssl_ft_in1k_224```: Pre-trained on a subset of YFCC100M using semi-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```swsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using semi-weakly supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet101```: ResNet-101.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet152```: ResNet-152.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ResNet-D

ResNet-D from _[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)_ by He et al. ResNet-D
is identical to ResNet, except that it supplants the 7 x 7 convolution in the stem with three 3 x 3 convolutions
and delegates spatial reduction in the identity branch of residual blocks to average pooling in lieu of a strided
1 x 1 convolution.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```resnet18d```: ResNet-D-18.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet34d```: ResNet-D-34.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet26d```: ResNet-D-26.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet50d```: ResNet-D-50.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet101d```: ResNet-D-101.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet152d```: ResNet-D-152.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet200d```: ResNet-D-200.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ResNet-T

ResNet-T from _[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)_ by Wightman. ResNet-T is identical to ResNet-D, except that the
stem is tiered, i.e., the output dimension of the convolutions are 24, 32, and 64 instead of 32, 32, and 64.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```resnet10t```: ResNet-T-10.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet14t```: ResNet-T-14.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnet26t```: ResNet-T-26.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## Wide ResNet

Wide ResNet from _[Wide Residual Networks](https://arxiv.org/abs/1605.07146)_ by Zagoruyko et al. Wide ResNet recommends widening the bottleneck dimension of residual blocks in ResNet as an alternative to deepening the model.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```wide_resnet50_2```: Wide ResNet-50, width multiplier 2x.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```wide_resnet101_2```: Wide ResNet-101, width multiplier 2x.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ResNeXt

ResNeXt from _[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)_ by Xie et al. ResNeXt is inspired by Inception's split-transform-merge strategy and has a multi-branch topology that aggregates a set of homogeneous transformations - 3 x 3 convolutions - in residual blocks for better performance.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```resnext50_32x4d```: ResNeXt-50, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 4.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```ssl_ft_in1k_224```: Pre-trained on a subset of YFCC100M using semi-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```swsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using semi-weakly supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnext101_32x4d```: ResNeXt-101, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 4.<br>
* ```ssl_ft_in1k_224```: Pre-trained on a subset of YFCC100M using semi-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```swsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using semi-weakly supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnext101_32x8d```: ResNeXt-101, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```ssl_ft_in1k_224```: Pre-trained on a subset of YFCC100M using semi-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```wsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using weakly-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```swsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using semi-weakly supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnext101_32x16d```: ResNeXt-101, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 16.<br>
* ```ssl_ft_in1k_224```: Pre-trained on a subset of YFCC100M using semi-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```wsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using weakly-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```swsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using semi-weakly supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnext101_32x32d```: ResNeXt-101, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 32.<br>
* ```wsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using weakly-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnext101_32x48d```: ResNeXt-101, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 48.<br>
* ```wsl_ft_in1k_224```: Pre-trained on 940 million Instagram images using weakly-supervised learning and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnext101_64x4d```: ResNeXt-101, cardinality 64, bottleneck dimensionality per cardinal group in the first stage 4.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## SENet

Squeeze-and-excitation network (SENet) from _[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)_ by Hu et al. SENet introduces squeeze-and-excitation (SE), a simple channel attention mechanism, for capturing relationships between different channels and adaptively weighing them.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```seresnet50```: SE-ResNet-50.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```seresnet152d```: SE-ResNet-D-152.<br>
* ```in1k_256```: Trained on ImageNet1K at resolution 256 x 256.

&#x25cf; ```seresnext26d_32x4d```: SE-ResNeXt-D-26, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 4.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```seresnext26t_32x4d```: SE-ResNeXt-T-26, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 4.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```seresnext50_32x4d```: SE-ResNeXt-50, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 4.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```seresnext101_32x8d```: SE-ResNeXt-101, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```seresnext101d_32x8d```: SE-ResNeXt-D-101, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ECANet

Efficient channel attention network (ECANet) from _[ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)_ by Wang et al. ECANet suggests an alternative to squeeze-and-excitation, dubbed efficient channel attention (ECA), that eliminates dimensionality reduction in the bottleneck layer of the excitation module
for better accuracy, in addition to being cheaper.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```ecaresnet26t```: ECA-ResNet-T-26.<br>
* ```in1k_256```: Trained on ImageNet1K at resolution 256 x 256.

&#x25cf; ```ecaresnet50_light```: Lightweight ECA-ResNet-50 where most of the layers are in stage 3.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```ecaresnet50d```: ECA-ResNet-D-50.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```ecaresnet50t```: ECA-ResNet-T-50.<br>
* ```in1k_256```: Trained on ImageNet1K at resolution 256 x 256.

&#x25cf; ```ecaresnet101d```: ECA-ResNet-D-101.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```ecaresnet269d```: ECA-ResNet-D-269.<br>
* ```in1k_320```: Trained on ImageNet1K at resolution 320 x 320.

## ResNet-RS

ResNet-RS from _[Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579)_ by Bello et al. ResNet-RS is architecturally simple, differing not much from SE-ResNet-D, but rivals the performance of more advanced networks like EfficientNet thanks to its enhanced training and scaling procedures.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnet.py).

&#x25cf; ```resnetrs50```: ResNet-RS-50.<br>
* ```in1k_160```: Trained on ImageNet1K at resolution 160 x 160.

&#x25cf; ```resnetrs101```: ResNet-RS-101.<br>
* ```in1k_192```: Trained on ImageNet1K at resolution 192 x 192.

&#x25cf; ```resnetrs152```: ResNet-RS-152.<br>
* ```in1k_256```: Trained on ImageNet1K at resolution 256 x 256.

&#x25cf; ```resnetrs200```: ResNet-RS-200.<br>
* ```in1k_256```: Trained on ImageNet1K at resolution 256 x 256.

&#x25cf; ```resnetrs270```: ResNet-RS-270.<br>
* ```in1k_256```: Trained on ImageNet1K at resolution 256 x 256.

&#x25cf; ```resnetrs350```: ResNet-RS-350.<br>
* ```in1k_288```: Trained on ImageNet1K at resolution 288 x 288.

&#x25cf; ```resnetrs420```: ResNet-RS-420.<br>
* ```in1k_320```: Trained on ImageNet1K at resolution 320 x 320.

## SKNet

Selective kernel networks (SKNet) from _[Selective Kernel Networks](https://arxiv.org/abs/1903.06586)_ by Li et al. SKNet uses selective kernel units (SK units), a branch attention mechanism, to adjust each residual block's
kernel size according to the input.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/sknet.py).

&#x25cf; ```skresnet18```: SK-ResNet-18.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```skresnet34```: SK-Resnet-34.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```skresnext50_32x4d```: SK-ResNeXt-50, cardinality 32, bottleneck dimensionality per cardinal group in the first stage 4.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ResNeSt

ResNeSt from _[ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)_ by Zhang et al. ResNeSt compounds the power of multi-branch architectures with that of channel attention
using a mechanism known as split attention (SplAt) that can be considered a generalization of SK but with static kernel sizes.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/resnest.py).

&#x25cf; ```resnest14_2s1x64d```: ResNeSt-14, radix 2, cardinality 1, dimensionality per radix group in the first stage 64.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnest26_2s1x64d```: ResNeSt-26, radix 2, cardinality 1, dimensionality per radix group in the first stage 64.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnest50_1s4x24d```: ResNeSt-50, radix 1, cardinality 4, dimensionality per radix group in the first stage 24.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnest50_2s1x64d```: ResNeSt-50, radix 2, cardinality 1, dimensionality per radix group in the first stage 64.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnest50_4s2x40d```: ResNeSt-50, radix 4, cardinality 2, dimensionality per radix group in the first stage 40.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```resnest101_2s1x64d```: ResNeSt-101, radix 2, cardinality 1, dimensionality per radix group in the first stage 64.<br>
* ```in1k_256```: Trained on ImageNet1K at resolution 256 x 256.

&#x25cf; ```resnest200_2s1x64d```: ResNeSt-200, radix 2, cardinality 1, dimensionality per radix group in the first stage 64.<br>
* ```in1k_320```: Trained on ImageNet1K at resolution 320 x 320.

&#x25cf; ```resnest269_2s1x64d```: ResNeSt-269, radix 2, cardinality 1, dimensionality per radix group in the first stage 64.<br>
* ```in1k_416```: Trained on ImageNet1K at resolution 416 x 416.

## Swin

Shifted window attention transformer (Swin transformer) from _[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)_ by Liu et al. Swin is a hierarchical vision transformer that efficiently calculates attention by limiting it to local windows and captures cross-window relationships using window shifting.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/swin_transformer.py).

&#x25cf; ```swin_tiny_window7```: Swin-Tiny, window size 7 x 7.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```swin_small_window7```: Swin-Small, window size 7 x 7.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```swin_base_window7```: Swin-Base, window size 7 x 7.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```swin_base_window12```: Swin-Base, window size 12 x 12.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```swin_large_window7```: Swin-Large, window size 7 x 7.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```swin_large_window12```: Swin-Large, window size 7 x 7.<br>
* ```in22k_224```: Trained on ImageNet22K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.

## Swin-S3

Swin-S3 from _[Searching the Search Space of Vision Transformer](https://arxiv.org/abs/2111.14725)_ by Chen et al. Swin-S3 is composed of the same type of blocks as Swin but was discovered through neural architecture search.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/swin_transformer.py).

&#x25cf; ```swin_s3_tiny_224```: Swin-S3-Tiny, resolution 224 x 224.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```swin_s3_small_224```: Swin-S3-Small, resolution 224 x 224.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```swin_s3_base_224```: Swin-S3-Base, resolution 224 x 224.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## VAN

Visual attention network (VAN) from _[Visual Attention Network](https://arxiv.org/abs/2202.09741)_ by Guo et al. VAN unifies the merits of convolutions and self-attention through large kernel attention (LKA), a cheap attention module with spatial and channel adaptability as well as the ability to capture long-range dependencies.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/Visual-Attention-Network/VAN-Classification).

&#x25cf; ```van_b0```: VAN-B0.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```van_b1```: VAN-B1.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```van_b2```: VAN-B2.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```van_b3```: VAN-B3.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## VGG

VGG from _[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)_ by Simonyan et al. VGG stacks many 3 x 3 convolutions on top of one another, interleaved with non-linearities, pooling, and potentially batch normalization, and was one of the first architectures exhibiting the importance of depth for vision tasks.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vgg.py).

&#x25cf; ```vgg11```: VGG-11.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```vgg11_bn```: VGG-11 with batch normalization.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```vgg13```: VGG-13.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```vgg13_bn```: VGG-13 with batch normalization.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```vgg16```: VGG-16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```vgg16_bn```: VGG-16 with batch normalization.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```vgg19```: VGG-19.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

&#x25cf; ```vgg19_bn```: VGG-19 with batch normalization.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.

## ViT

Vision transformer (ViT) from _[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)_ by Dosovitskiy et al. ViT closely resembles NLP transformers but the input is first patchified and linearly transformed to manage the quadratic complexity of self-attention.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py).

&#x25cf; ```vit_tiny_patch16```: ViT-Tiny, patch size 16 x 16.<br>
* ```augreg_in22k_224```: Trained on ImageNet22K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_384```: Trained on ImageNet22K with additional augmentation & regularization at resolution 384 x 384.

&#x25cf; ```vit_small_patch32```: ViT-Small, patch size 32 x 32.<br>
* ```augreg_in22k_224```: Trained on ImageNet22K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_384```: Trained on ImageNet22K with additional augmentation & regularization at resolution 384 x 384.

&#x25cf; ```vit_small_patch16```: ViT-Small, patch size 16 x 16.<br>
* ```dino_in1k_224```: Trained using DINO on ImageNet1K at resolution 224 x 224.<br>
* ```augreg_in22k_224```: Trained on ImageNet22K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_384```: Trained on ImageNet22K with additional augmentation & regularization at resolution 384 x 384.

&#x25cf; ```vit_small_patch8```: ViT-Small, patch size 8 x 8.<br>
* ```dino_in1k_224```: Trained using DINO on ImageNet1K at resolution 224 x 224.<br>

&#x25cf; ```vit_base_patch32```: ViT-Base, patch size 32 x 32.<br>
* ```sam_in1k_224```: Trained using SAM on ImageNet1K at resolution 224 x 224.<br>
* ```augreg_in22k_224```: Trained on ImageNet22K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_384```: Trained on ImageNet22K with additional augmentation & regularization at resolution 384 x 384.

&#x25cf; ```vit_base_patch16```: ViT-Base, patch size 16 x 16.<br>
* ```sam_in1k_224```: Trained using SAM on ImageNet1K at resolution 224 x 224.<br>
* ```dino_in1k_224```: Trained using DINO on ImageNet1K at resolution 224 x 224.<br>
* ```augreg_in22k_224```: Trained on ImageNet22K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_384```: Trained on ImageNet22K with additional augmentation & regularization at resolution 384 x 384.

&#x25cf; ```vit_base_patch8```: ViT-Base, patch size 8 x 8.<br>
* ```dino_in1k_224```: Trained using DINO on ImageNet1K at resolution 224 x 224.<br>
* ```augreg_in22k_224```: Trained on ImageNet22K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K with additional augmentation & regularization at resolution 224 x 224.<br>

&#x25cf; ```vit_large_patch32```: ViT-Large, patch size 32 x 32.<br>
* ```orig_in22k_224```: Trained on ImageNet22K (original weights) at resolution 224 x 224.<br>
* ```orig_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K (original weights) at resolution 224 x 224.<br>

&#x25cf; ```vit_large_patch16```: ViT-Large, patch size 16 x 16.<br>
* ```augreg_in22k_224```: Trained on ImageNet22K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K with additional augmentation & regularization at resolution 224 x 224.<br>
* ```augreg_in22k_ft_in1k_384```: Trained on ImageNet22K with additional augmentation & regularization at resolution 384 x 384.

&#x25cf; ```vit_huge_patch14```: ViT-Huge, patch size 14 x 14.<br>
* ```orig_in22k_224```: Trained on ImageNet22K (original weights) at resolution 224 x 224.<br>

&#x25cf; ```vit_base_clip_patch32```: ViT-Base with an extra layer normalization before the transformer layers, patch size 32 x 32.<br>
* ```clip_openai_224```: Trained using CLIP on 400 million (image, text) pairs at resolution 224 x 224.<br>
* ```clip_openai_ft_in1k_224```: Pre-trained using CLIP on 400 million (image, text) pairs and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```clip_laion2b_224```: Trained using CLIP on LAION-2B at resolution 224 x 224.<br>
* ```clip_laion2b_ft_in1k_224```: Pre-trained using CLIP on LAION-2B and fine-tuned on ImageNet1K at resolution 224 x 224.<br>

&#x25cf; ```vit_base_clip_patch16```: ViT-Base with an extra layer normalization before the transformer layers, patch size 16 x 16.<br>
* ```clip_openai_224```: Trained using CLIP on 400 million (image, text) pairs at resolution 224 x 224.<br>
* ```clip_openai_ft_in1k_224```: Pre-trained using CLIP on 400 million (image, text) pairs and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```clip_openai_ft_in1k_384```: Pre-trained using CLIP on 400 million (image, text) pairs and fine-tuned on ImageNet1K at resolution 384 x 384.<br>
* ```clip_laion2b_224```: Trained using CLIP on LAION-2B at resolution 224 x 224.<br>
* ```clip_laion2b_ft_in1k_224```: Pre-trained using CLIP on LAION-2B and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```clip_laion2b_ft_in1k_384```: Pre-trained using CLIP on LAION-2B and fine-tuned on ImageNet1K at resolution 384 x 384.<br>

&#x25cf; ```vit_large_clip_patch14```: ViT-Large with an extra layer normalization before the transformer layers, patch size 14 x 14.<br>
* ```clip_openai_224```: Trained using CLIP on 400 million (image, text) pairs at resolution 224 x 224.<br>
* ```clip_openai_ft_in1k_224```: Pre-trained using CLIP on 400 million (image, text) pairs and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```clip_laion2b_224```: Trained using CLIP on LAION-2B at resolution 224 x 224.<br>
* ```clip_laion2b_ft_in1k_224```: Pre-trained using CLIP on LAION-2B and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```clip_laion2b_ft_in1k_336```: Pre-trained using CLIP on LAION-2B and fine-tuned on ImageNet1K at resolution 336 x 336.<br>

&#x25cf; ```vit_huge_clip_patch14```: ViT-Huge with an extra layer normalization before the transformer layers, patch size 14 x 14.<br>
* ```clip_laion2b_224```: Trained using CLIP on LAION-2B at resolution 224 x 224.<br>
* ```clip_laion2b_ft_in1k_224```: Pre-trained using CLIP on LAION-2B and fine-tuned on ImageNet1K at resolution 224 x 224.<br>

```vit_giant_clip_patch14```: ViT-Giant with an extra layer normalization before the transformer layers, patch size 14 x 14.<br>
* ```clip_laion2b_224```: Trained using CLIP on LAION-2B at resolution 224 x 224.<br>

## DeiT 3

Data-efficient image transformer III (DeiT 3) from _[DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118)_ by Touvron et al. DeiT 3 is architecturally similar to ViT but is trained using a refined training recipe.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/deit.py).

&#x25cf; ```deit3_small_patch16```: DeiT3-Small, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```deit3_medium_patch16```: DeiT3-Medium, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.

&#x25cf; ```deit3_base_patch16```: DeiT3-Base, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```deit3_large_patch16```: DeiT3-Large, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in1k_384```: Trained on ImageNet1K at resolution 384 x 384.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_384```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 384 x 384.

&#x25cf; ```deit3_huge_patch14_224```: DeiT3-Huge, patch size 14 x 14.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```in22k_ft_in1k_224```: Pre-trained on ImageNet22K and fine-tuned on ImageNet1K at resolution 224 x 224.

## BEiT

BEiT from _[BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)_ by Bao et al. BEiT is architecturally similar to ViT but is trained using BERT-style masked modelling.
<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/beit.py).

&#x25cf; ```beit_base_patch16```: BEiT-Base, patch size 16 x 16.<br>
* ```beit_in22k_ft_in22k_224```: Pre-trained using BeiT on ImageNet22K and fine-tuned with labels on ImageNet22K at resolution 224 x 224.<br>
* ```beit_in22k_ft_in22k_ft_in1k_224```: Pre-trained using BeiT on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```beit_in22k_ft_in22k_ft_in1k_224```: Pre-trained using BeiT on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 384 x 384.<br>
* ```beitv2_in1k_ft_in22k_224```: Pre-trained using BeiT V2 on ImageNet1K and fine-tuned with labels on ImageNet22K at resolution 224 x 224.<br>
* ```beitv2_in1k_ft_in22k_ft_in1k_224```: Pre-trained using BeiT V2 on ImageNet1K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.

&#x25cf; ```beit_large_patch16```: BEiT-Large, patch size 16 x 16.<br>
* ```beit_in22k_ft_in22k_224```: Pre-trained using BeiT on ImageNet22K and fine-tuned with labels on ImageNet22K at resolution 224 x 224.<br>
* ```beit_in22k_ft_in22k_ft_in1k_224```: Pre-trained using BeiT on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.<br>
* ```beit_in22k_ft_in22k_ft_in1k_224```: Pre-trained using BeiT on ImageNet22K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 384 x 384.<br>
* ```beitv2_in1k_ft_in22k_224```: Pre-trained using BeiT V2 on ImageNet1K and fine-tuned with labels on ImageNet22K at resolution 224 x 224.<br>
* ```beitv2_in1k_ft_in22k_ft_in1k_224```: Pre-trained using BeiT V2 on ImageNet1K, fine-tuned with labels on ImageNet22K,
and fine-tuned with labels on ImageNet1K at resolution 224 x 224.

## XCiT

Cross-covariance image transformer (XCiT) from _[XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)_ by El-Nouby et al. XCiT
eliminates the quadratic complexity of traditional transformers using cross-covariance attention (XCA), an inverted version of self-attention that is applied along the feature rather than the token axis of the input, and complements it with local patch interaction modules (LPI) that explicitly enable communications amongst patches for scalable, efficient image transformers.<br>
For the reference implementation, source of pre-trained parameters, and copyrights,
please visit [here](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/xcit.py).

&#x25cf; ```xcit_nano12_patch16```: XCiT-Nano, depth 12, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf; ```xcit_nano12_patch8```: XCiT-Nano, depth 12, patch size 8 x 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf; ```xcit_tiny12_patch16```: XCiT-Tiny, depth 12, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf; ```xcit_tiny12_patch8```: XCiT-Tiny, depth 12, patch size 8 x 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf; ```xcit_tiny24_patch16```: XCiT-Tiny, depth 24, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_tiny24_patch8```: XCiT-Tiny, depth 24, patch size 8 x 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_small12_patch16```: XCiT-Small, depth 12, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_small12_patch8```: XCiT-Small, depth 12, patch size 8 x 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_small24_patch16```: XCiT-Small, depth 24, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_small24_patch8```: XCiT-Small, depth 24, patch size 8 x 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_medium24_patch16```: XCiT-Medium, depth 24, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_medium24_patch8```: XCiT-Medium, depth 24, patch size 8 x 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_large24_patch16```: XCiT-Large, depth 24, patch size 16 x 16.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.

&#x25cf;  ```xcit_large24_patch8```: XCiT-Large, depth 24, patch size 8 x 8.<br>
* ```in1k_224```: Trained on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_224```: Trained using distillation on ImageNet1K at resolution 224 x 224.<br>
* ```dist_in1k_384```: Trained using distillation on ImageNet1K at resolution 384 x 384.
