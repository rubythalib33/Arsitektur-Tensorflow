import tensorflow as tf
import numpy as np
#VGG
VGG = {}
VGG['VGG16'] = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512]
VGG['VGG19'] = [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512]

class VGGNet(tf.keras.Model):
    def __init__(self, num_class=1000, VGG_type='VGG16'):
        super(VGGNet, self).__init__()
        self.vgg = tf.keras.Sequential()
        
        for a in VGG[VGG_type]:
            if a == 'M':
                self.vgg.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
            else:
                self.vgg.add(tf.keras.layers.Conv2D(a,3,padding="same",activation='relu'))
        self.vgg.add(tf.keras.layers.Flatten())
        self.vgg.add(tf.keras.layers.Dense(4096,activation='relu'))
        self.vgg.add(tf.keras.layers.Dense(4096,activation='relu'))
        self.vgg.add(tf.keras.layers.Dense(num_class,activation='softmax'))
    def call(self,x):
        return self.vgg(x)


if __name__ == '__main__':
    image = tf.random.normal((1,572,572,1))
    model = VGGNet(num_class=5)
    out = model(image)
    print(out.shape)
