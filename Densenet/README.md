# DenseNet
In this directory, I will implement DenseNet according to official Model 
by TensorFlow. And I will explain code details and the paper.

## DenseNet Model
In this module, I will explain the paper with code details.

### Params

- densenet_size: A single integer for the size of the DenseNet model.It should be 3n+4
- num_classes: The number of classes used as labels.
- densenet_version:string representing which version of the DenseNet network to use .Valid values:
['BC','Basic']
- reduction:A single float for reduce rate of the transition layer
- drop_rate: A single float for drop_out
- growth_rate: A single integer for block layer to keep dimension
- data_name: Cifar-10 or ImageNet
- data_format:Input format ('channels_last', 'channels_first', or None).
If set to None, the format is dependent on whether a GPU is available.
- change_dataformat_NCHW:Use data_format NCHW or NHWC.
- dtype:TensorFlow dtype to use for calculations. If not specified
tf.float32 is used.

### Structure
1.Building block layer

The BN-ReLU-Conv(3x3) version of Hx

2.Bottleneck layers.
 
A 1x1 convolution can be introduced as bottleneck layer before each 3×3 convolution
to reduce the number of input feature-maps, and thus to improve computational efficiency.

The BN-ReLU-Conv(1x1)-BN-Relu-Conv(3x3) version of Hx,let each
1x1 convolution produce 4k feature-maps.

3.Block layer

params:
- inputs:
- nb_layers:the number of block_layer
- growth_rate: a single integer representing dimension after block_fn 
- block_fn:bottleneck_layer or building_block
- drop_rate:for drop_rate
- training: Either True of False,whether we are currently training the model.
- name:A string name for the tensor output of the block layer
- data_format: channels_first or channels_last

4.Transition layer

Reduce the number of feature-maps at transition
layers. If a dense block contains m feature-maps, we let
the following transition layer generate bθmc output feature-maps,
where 0 <θ ≤1 is referred to as the compression factor.
When θ=1, the number of feature-maps across transition layers remains unchanged


## cifar10_main.py

### params

- data_dir
- model_dir
- mode
- data_format
- clean
- clean
- train_epochs
- stop_threshold
- export_dir
- batch_size
- train_epochs
- densenet_version

See more code details in cifar10_main.py