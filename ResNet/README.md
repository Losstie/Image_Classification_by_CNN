# ResNet
In this directory, I will implement ResNet according to official Model 
by TensorFlow. And I will explain code details and the paper.

## Resnet Model
In this module, I will explain the paper with code details.

### Params
- resnet_size: A single integer for the size of the ResNet model.It should be 6n+2.
- bottleneck: Use regular blocks or bottleneck blocks.
- num_classes:The number of classes used as labels.
- num_filters:The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
- kernel_size:The kernel size to use for convolution.
- conv_stride:stride size for the initial convolutional layer
- first_pool_size:Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
- first_pool_stride:stride size for the first pooling layer. Not used
            if first_pool_size is None.
- block_size:A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
- block_stride:List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
- resnet_version:Integer representing which version of the ResNet network
            to use.Valid values: [1, 2]
- data_format:Input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
- dtype:The TensorFlow dtype to use for calculations. If not specified
            tf.float32 is used.


### Structure
The block layer's has two kinds of structures, which is regular and bottleneck structure.The regular block layer has one conv_layer,and bottleneck has three.

** As shown below,this is a building block of Residual learning.

<img src="https://github.com/Losstie/Image_Classification_by_CNN/blob/master/ResNet/images/building_block.png" alt="building_block" />

A deeper residual function F for ImageNet. Left: a building block (on 56×56 feature maps) as in Fig. for ResNet-34.
Right: a “bottleneck” building block for ResNet-50/101/152. 

<img src="https://github.com/Losstie/Image_Classification_by_CNN/blob/master/ResNet/images/block_layer.png" alt="block_layer" />

#### Version_1
1.regular 

Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

2.bottleneck

Similar to regular, except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

#### Version_2

1.regular

Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

2.bottleneck

described in:
        Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
        
 Adapted to the ordering conventions of:
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

## Resnet_run.py
see more in future