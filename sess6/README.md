# Normalization and Regularization Experiments

## Normalization Methods

Three main methods are currently used: Batch Normalization, Group Normalization, and Layer Normalization

### Batch Normalization

![Image](image.png "Image")
![BatchNorm](batchnorm.png "BatchNorm")

### Group Normalization
![Image](image.png "Image")
![GroupNorm](groupnorm.png "GroupNorm")
### Layer Normalization
![Image](image.png "Image")
![LayerNorm](layernorm.png "LayerNorm")
## Training Experiments

### Setup

BatchNorm, GroupNorm, and LayerNorm are supported by the model - usage of each is through passing params for each.

For BatchNorm and LayerNorm, default Pytorch values are used.
For GroupNorm, group size of `2` is used.

For L1 regularization, lambda of `0.01` is used.
For L2 regularization, same value, `0.01` of weight decay is used.

LR - `0.0` is kept the same throughout.
Model is trained for `20` epochs, `SGD` optimizer.

### Results

![BatchNorm](bnAll.png "BatchNorm - simple, L1, L2, L1+L2")
![BatchNorm, GroupNorm, LayerNorm](bnGnLn.png "BatchNorm, GroupNorm, LayerNorm")
![BatchNorm+L1, GroupNorm, LayerNorm](bnl1GnLn.png "BatchNorm+L1, GroupNorm, LayerNorm")
![BatchNorm+L1+L2, GroupNorm+L1, LayerNorm+L2](bnAll.png "BatchNorm+L1+L2, GroupNorm+L1, LayerNorm+L2")


### Inference

Full training logs + misclassified samples & graphs in the `Normalization.ipynb` notebook