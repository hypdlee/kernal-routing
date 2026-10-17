# Compute estimate (ESTIMATE, not a measurement)

MACs are counted per image, forward pass. Hours scale the ResNet-18 study's logged
minutes by the MAC ratio for the same cell. Depthwise-heavy models (MobileNetV2,
ConvNeXt-Tiny) are memory bound and will be slower than this. The runner's probe
measures the real rate before each group and prints a measured ETA.

| model | CIFAR-100 w64 static GMACs | est. GPU-h, 132 runs | vs ResNet-18 |
|---|---|---|---|
| resnet18 | 0.555 | 204 | 1.00x |
| mobilenetv2 | 0.088 | 35 | 0.17x |
| vgg16 | 0.313 | 121 | 0.59x |
| resnet50 | 1.298 | 467 | 2.29x |
| convnext_tiny | 1.450 | 518 | 2.54x |

Four new models together: 1141 GPU-h (estimate).

## MACs per image, every cell

| model | dataset | width | K | arm | MMACs |
|---|---|---|---|---|---|
| convnext_tiny | cifar100 | 16 | 4 | gap | 97.5 |
| convnext_tiny | cifar100 | 16 | 4 | static | 97.1 |
| convnext_tiny | cifar100 | 16 | 4 | uniform | 97.5 |
| convnext_tiny | cifar100 | 32 | 2 | gap | 371.8 |
| convnext_tiny | cifar100 | 32 | 4 | gap | 372.1 |
| convnext_tiny | cifar100 | 32 | 4 | static | 371.2 |
| convnext_tiny | cifar100 | 32 | 4 | uniform | 371.9 |
| convnext_tiny | cifar100 | 32 | 8 | gap | 372.7 |
| convnext_tiny | cifar100 | 64 | 4 | frozen_random | 1451.6 |
| convnext_tiny | cifar100 | 64 | 4 | gap | 1452.4 |
| convnext_tiny | cifar100 | 64 | 4 | shuffled | 1452.4 |
| convnext_tiny | cifar100 | 64 | 4 | static | 1450.3 |
| convnext_tiny | cifar100 | 64 | 4 | uniform | 1451.6 |
| convnext_tiny | tinyimagenet | 32 | 4 | gap | 1485.7 |
| convnext_tiny | tinyimagenet | 32 | 4 | static | 1484.9 |
| convnext_tiny | tinyimagenet | 32 | 4 | uniform | 1485.5 |
| convnext_tiny | tinyimagenet | 64 | 4 | gap | 5803.0 |
| convnext_tiny | tinyimagenet | 64 | 4 | static | 5800.9 |
| convnext_tiny | tinyimagenet | 64 | 4 | uniform | 5802.2 |
| mobilenetv2 | cifar100 | 16 | 4 | gap | 9.7 |
| mobilenetv2 | cifar100 | 16 | 4 | static | 9.6 |
| mobilenetv2 | cifar100 | 16 | 4 | uniform | 9.7 |
| mobilenetv2 | cifar100 | 32 | 2 | gap | 28.4 |
| mobilenetv2 | cifar100 | 32 | 4 | gap | 28.5 |
| mobilenetv2 | cifar100 | 32 | 4 | static | 28.1 |
| mobilenetv2 | cifar100 | 32 | 4 | uniform | 28.2 |
| mobilenetv2 | cifar100 | 32 | 8 | gap | 28.6 |
| mobilenetv2 | cifar100 | 64 | 4 | frozen_random | 88.3 |
| mobilenetv2 | cifar100 | 64 | 4 | gap | 89.5 |
| mobilenetv2 | cifar100 | 64 | 4 | shuffled | 89.5 |
| mobilenetv2 | cifar100 | 64 | 4 | static | 88.1 |
| mobilenetv2 | cifar100 | 64 | 4 | uniform | 88.3 |
| mobilenetv2 | tinyimagenet | 32 | 4 | gap | 112.5 |
| mobilenetv2 | tinyimagenet | 32 | 4 | static | 112.1 |
| mobilenetv2 | tinyimagenet | 32 | 4 | uniform | 112.2 |
| mobilenetv2 | tinyimagenet | 64 | 4 | gap | 353.5 |
| mobilenetv2 | tinyimagenet | 64 | 4 | static | 352.1 |
| mobilenetv2 | tinyimagenet | 64 | 4 | uniform | 352.4 |
| resnet18 | cifar100 | 16 | 4 | gap | 37.8 |
| resnet18 | cifar100 | 16 | 4 | static | 35.1 |
| resnet18 | cifar100 | 16 | 4 | uniform | 37.8 |
| resnet18 | cifar100 | 32 | 2 | gap | 144.9 |
| resnet18 | cifar100 | 32 | 4 | gap | 150.4 |
| resnet18 | cifar100 | 32 | 4 | static | 139.3 |
| resnet18 | cifar100 | 32 | 4 | uniform | 150.3 |
| resnet18 | cifar100 | 32 | 8 | gap | 161.4 |
| resnet18 | cifar100 | 64 | 4 | frozen_random | 599.4 |
| resnet18 | cifar100 | 64 | 4 | gap | 599.7 |
| resnet18 | cifar100 | 64 | 4 | shuffled | 599.7 |
| resnet18 | cifar100 | 64 | 4 | static | 555.5 |
| resnet18 | cifar100 | 64 | 4 | uniform | 599.4 |
| resnet18 | tinyimagenet | 32 | 4 | gap | 568.3 |
| resnet18 | tinyimagenet | 32 | 4 | static | 557.2 |
| resnet18 | tinyimagenet | 32 | 4 | uniform | 568.2 |
| resnet18 | tinyimagenet | 64 | 4 | gap | 2266.0 |
| resnet18 | tinyimagenet | 64 | 4 | static | 2221.8 |
| resnet18 | tinyimagenet | 64 | 4 | uniform | 2265.7 |
| resnet50 | cifar100 | 16 | 4 | gap | 84.3 |
| resnet50 | cifar100 | 16 | 4 | static | 81.5 |
| resnet50 | cifar100 | 16 | 4 | uniform | 84.3 |
| resnet50 | cifar100 | 32 | 2 | gap | 330.7 |
| resnet50 | cifar100 | 32 | 4 | gap | 336.4 |
| resnet50 | cifar100 | 32 | 4 | static | 325.0 |
| resnet50 | cifar100 | 32 | 4 | uniform | 336.3 |
| resnet50 | cifar100 | 32 | 8 | gap | 347.7 |
| resnet50 | cifar100 | 64 | 4 | frozen_random | 1343.3 |
| resnet50 | cifar100 | 64 | 4 | gap | 1343.6 |
| resnet50 | cifar100 | 64 | 4 | shuffled | 1343.6 |
| resnet50 | cifar100 | 64 | 4 | static | 1298.0 |
| resnet50 | cifar100 | 64 | 4 | uniform | 1343.3 |
| resnet50 | tinyimagenet | 32 | 4 | gap | 1311.2 |
| resnet50 | tinyimagenet | 32 | 4 | static | 1299.8 |
| resnet50 | tinyimagenet | 32 | 4 | uniform | 1311.1 |
| resnet50 | tinyimagenet | 64 | 4 | gap | 5237.2 |
| resnet50 | tinyimagenet | 64 | 4 | static | 5191.6 |
| resnet50 | tinyimagenet | 64 | 4 | uniform | 5236.9 |
| vgg16 | cifar100 | 16 | 4 | gap | 23.6 |
| vgg16 | cifar100 | 16 | 4 | static | 19.9 |
| vgg16 | cifar100 | 16 | 4 | uniform | 23.6 |
| vgg16 | cifar100 | 32 | 2 | gap | 86.2 |
| vgg16 | cifar100 | 32 | 4 | gap | 93.6 |
| vgg16 | cifar100 | 32 | 4 | static | 78.8 |
| vgg16 | cifar100 | 32 | 4 | uniform | 93.5 |
| vgg16 | cifar100 | 32 | 8 | gap | 108.3 |
| vgg16 | cifar100 | 64 | 4 | frozen_random | 372.1 |
| vgg16 | cifar100 | 64 | 4 | gap | 372.5 |
| vgg16 | cifar100 | 64 | 4 | shuffled | 372.5 |
| vgg16 | cifar100 | 64 | 4 | static | 313.2 |
| vgg16 | cifar100 | 64 | 4 | uniform | 372.1 |
| vgg16 | tinyimagenet | 32 | 4 | gap | 329.8 |
| vgg16 | tinyimagenet | 32 | 4 | static | 315.0 |
| vgg16 | tinyimagenet | 32 | 4 | uniform | 329.7 |
| vgg16 | tinyimagenet | 64 | 4 | gap | 1312.1 |
| vgg16 | tinyimagenet | 64 | 4 | static | 1252.9 |
| vgg16 | tinyimagenet | 64 | 4 | uniform | 1311.7 |
