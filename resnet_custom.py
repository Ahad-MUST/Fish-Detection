# resnet_custom_tf.py
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Please install it using:")
    print("pip install tensorflow")
    exit(1)


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channels, stride=1, downsample=None, dropout_prob=0.3, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_prob = dropout_prob
        
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, strides=stride,
                                  padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(dropout_prob)

        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                  padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample

    def call(self, x, training=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.dropout(out, training=training)  # Dropout after first ReLU

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out = layers.Add()([out, identity])
        out = self.relu(out)

        return out


class DownsampleBlock(layers.Layer):
    def __init__(self, out_channels, stride, **kwargs):
        super(DownsampleBlock, self).__init__(**kwargs)
        self.conv = layers.Conv2D(out_channels, kernel_size=1, strides=stride, use_bias=False)
        self.bn = layers.BatchNormalization()
    
    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


class CustomResNet(keras.Model):
    def __init__(self, block, layers_config, num_classes=3, dropout_prob=0.5, **kwargs):
        super(CustomResNet, self).__init__(**kwargs)
        self.in_channels = 64
        self.dropout_prob = dropout_prob

        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self._make_layer(block, 64, layers_config[0])
        self.layer2 = self._make_layer(block, 128, layers_config[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers_config[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers_config[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(self.dropout_prob)  # Dropout before FC
        self.fc = layers.Dense(num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = DownsampleBlock(out_channels * block.expansion, stride)

        layer_list = []
        layer_list.append(block(out_channels, stride, downsample, dropout_prob=self.dropout_prob))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layer_list.append(block(out_channels, dropout_prob=self.dropout_prob))

        return keras.Sequential(layer_list)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x, training=training)  # Early dropout

        x = self.layer1(x, training=training)
        x = self.dropout(x, training=training)  # Dropout after layer1
        x = self.layer2(x, training=training)

        x = self.avgpool(x)
        x = self.dropout(x, training=training)  # Dropout before FC
        x = self.fc(x)

        return x


def custom_resnet18(num_classes=3, dropout_prob=0.5):
    return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_prob=dropout_prob)


# Helper function to create model with proper input shape
def create_custom_resnet18(input_shape=(512, 512, 3), num_classes=3, dropout_prob=0.5):
    inputs = keras.Input(shape=input_shape)
    model = custom_resnet18(num_classes=num_classes, dropout_prob=dropout_prob)
    outputs = model(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)