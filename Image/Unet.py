import tensorflow as tf
import numpy as np

def double_conv(out_c):
    conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(out_c, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(out_c, kernel_size=3, activation='relu')
    ])
    return conv

def crop_image(tensor, target_tensor):
    target_size = target_tensor.shape[2]
    tensor_size = tensor.shape[2]
    delta = tensor_size - target_size

    delta = delta//2
    return tensor[:,delta:tensor_size-delta,delta:tensor_size-delta,:]

class UNet(tf.keras.Model):
    def __init__(self, num_class):
        super(UNet, self).__init__()

        self.maxPool2D = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)
        self.down_conv_1 = double_conv(64)
        self.down_conv_2 = double_conv(128)
        self.down_conv_3 = double_conv(256)
        self.down_conv_4 = double_conv(512)
        self.down_conv_5 = double_conv(1024)

        self.up_trans_1 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2)
        self.up_conv_1 = double_conv(512)

        self.up_trans_2 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2)
        self.up_conv_2 = double_conv(256)

        self.up_trans_3 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2)
        self.up_conv_3 = double_conv(128)

        self.up_trans_3 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2)
        self.up_conv_3 = double_conv(64)

        self.last_out = tf.keras.layers.Conv2D(num_class+1, kernel_size=1)

    def call(self, x):
        #encoder
        x1 = self.down_conv_1(x)
        x2 = self.maxPool2D(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.maxPool2D(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.maxPool2D(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.maxPool2D(x7)
        x9 = self.down_conv_5(x8)

        #decoder
        x = self.up_trans_1(x9)
        print(x.shape)
        y = crop_image(x7,x)
        print(y.shape)
        x = self.up_conv_1(tf.concat([y,x],3))
        print(x.shape)

        x = self.up_trans_2(x)
        y = crop_image(x5,x)
        x = self.up_conv_2(tf.concat([y,x],3))

        x = self.up_trans_3(x)
        y = crop_image(x3,x)
        x = self.up_conv_3(tf.concat([y,x],3))

        x = self.last_out(x)

        return x
if __name__ == "__main__":
    image = tf.random.normal((1,572,572,1))
    model = UNet(2)
    out = model(image)
    print(out.shape)
