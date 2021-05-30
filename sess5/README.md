# MNIST Small Model

The goal is to build a model to achieve 99.4% accuracy on the MNIST test set, trained for 15 epochs, with the model having < 10k params.

Only simple Conv2D layers are used for Convolutions.

## Iteration-1

The first hypothesis taken is that even though images size are small, 28x28, it should still be possible to have 2 Max Pooling layers in the network - given that MNIST is a relatively easier task.

For the first network, we put the first Max Pool layer after a Receptive Field(RF) of 5x5, followed by a Conv, followed by another Max Pool. At this point, we are at RF of 24. Adding another Conv layer, RF now is at 26, where the layers are stopped, and Avg Pool followed by 1x1 Conv is added to get the result.

Kernel sizes are kept same at 16.

The goal here is to train a small network that is simple and achieves a decent accuracy.

No image augmentations, dropout is done. 

SGD with Constant LR of 0.01 is used while training.

### Network Description

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
            Conv2d-4           [-1, 16, 24, 24]           2,304
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
         MaxPool2d-7           [-1, 16, 12, 12]               0
            Conv2d-8           [-1, 16, 10, 10]           2,304
              ReLU-9           [-1, 16, 10, 10]               0
      BatchNorm2d-10           [-1, 16, 10, 10]              32
        MaxPool2d-11             [-1, 16, 5, 5]               0
           Conv2d-12             [-1, 16, 3, 3]           2,304
             ReLU-13             [-1, 16, 3, 3]               0
      BatchNorm2d-14             [-1, 16, 3, 3]              32
        AvgPool2d-15             [-1, 16, 1, 1]               0
           Conv2d-16             [-1, 10, 1, 1]             160
================================================================
Total params: 7,344
Trainable params: 7,344
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.52
Params size (MB): 0.03
Estimated Total Size (MB): 0.55
----------------------------------------------------------------

### Results
1. Parameters: 7.3k
2. Best Train accuracy: `99.59` (15th epoch)
3. Best Test accuracy: `98.99` (9th epoch), `98.93` (14th epoch)

### Analysis
1. The network satisfies the size limit reqd, but the test accuracy is not in the desired range.
2. 99.59 train accuracy suggests that the base network is good and has capacity to improve.
3. Network looks to be overfitting at this point. Augmentation, regularizations should close this gap.


## Iteration-2

Resuming from the previous model, the goal is to now hit the test accuracy requirement, even if it means having a slightly larger model.

To be on the safe side and make sure we hit our accuracy requirement, model size is increased a little bit. Instead of using a GAP layer, we use a 3x3 Convolution at the end - this can probably be removed in the next step if we want to reduce the size again.

Since the model was overfitting, we add 2 dropout layers (0.1 value), 1 each after the MaxPool layers - since maxpool has reduced the dimension a lot, we want to further make it tougher for the model so adding a dropout here.

Apart from this, testing out different augmentation strategies, we settle on having random rotation of `-6,6` & a translate of `0.1, 0.1`.

Other training params remain the same.

### Network Description

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
            Conv2d-4           [-1, 16, 24, 24]           2,304
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
         MaxPool2d-7           [-1, 16, 12, 12]               0
           Dropout-8           [-1, 16, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           2,304
             ReLU-10           [-1, 16, 10, 10]               0
      BatchNorm2d-11           [-1, 16, 10, 10]              32
        MaxPool2d-12             [-1, 16, 5, 5]               0
          Dropout-13             [-1, 16, 5, 5]               0
           Conv2d-14             [-1, 16, 3, 3]           2,304
             ReLU-15             [-1, 16, 3, 3]               0
      BatchNorm2d-16             [-1, 16, 3, 3]              32
           Conv2d-17             [-1, 10, 1, 1]           1,440
================================================================
Total params: 8,624
Trainable params: 8,624
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.54
Params size (MB): 0.03
Estimated Total Size (MB): 0.58
----------------------------------------------------------------

### Results
1. Parameters: **8.6k**
2. Best Train accuracy: `98.45` (15th epoch)
3. Best Test accuracy: `**99.44**` (14th epoch), `99.42` (13th epoch)

### Analysis
1. We hit our test accuracy req, and the size of the model is < 10k, so all our objectives are met.
2. This model is fairly well regularised, as the test accuracy is better than the train throughout the training.
3. Since the train accuracy isn't at the level of previous iteration, it should be possible to squeeze out better results from this model, with better LR tuning.


## Iteration-3

We already reached our objective in the previous iteration. Here, we further try to reduce the model size to keep it below < 8k, while having the same test accuracy.

We keep the same architecture as before, only change the 1st layer's kernel size from 16 -> 8.
Same augmentations are used.

While trying out different LR to train with, we settle on 0.1, instead of 0.01 of the previous step.
Further, while checking the train graphs, we found 2 points to reduce the LR for the model to learn better.
We use a MultiStepLR scheduler, to reduce the LR by a factor of 0.1, after epochs 6 & 13.

### Network Description

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
            Conv2d-4           [-1, 16, 24, 24]           1,152
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
         MaxPool2d-7           [-1, 16, 12, 12]               0
           Dropout-8           [-1, 16, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           2,304
             ReLU-10           [-1, 16, 10, 10]               0
      BatchNorm2d-11           [-1, 16, 10, 10]              32
        MaxPool2d-12             [-1, 16, 5, 5]               0
          Dropout-13             [-1, 16, 5, 5]               0
           Conv2d-14             [-1, 16, 3, 3]           2,304
             ReLU-15             [-1, 16, 3, 3]               0
      BatchNorm2d-16             [-1, 16, 3, 3]              32
           Conv2d-17             [-1, 10, 1, 1]           1,440
================================================================
Total params: 7,384
Trainable params: 7,384
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.42
Params size (MB): 0.03
Estimated Total Size (MB): 0.45
----------------------------------------------------------------

### Results
1. Parameters: **7.38k**
2. Best Train accuracy: `98.6` (14th epoch)
3. Best Test accuracy: `**99.4**` (14th epoch), `99.39` (15th epoch)

### Analysis
1. We further reduce the model size to < 8k, while achieving the required test accuracy. However, this training is not as stable as the previous iteration. The model just touches 99.4, but doesn't hit that mark regularly.
2. It's still a fairly well regularised model, training acc is only `98.6`, so model has the capacity to learn more, possibly with more LR experiments.