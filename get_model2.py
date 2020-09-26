from tensorflow.keras.applications import MobileNet
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np

def get_mobilenet_SSD(image_size, num_classes):
    mobilenet = MobileNet(input_shape=image_size, include_top=False, weights="imagenet")
    for layer in mobilenet.layers:
        layer._name = layer.name + '_base'

    x = layers.BatchNormalization(beta_initializer='glorot_uniform', gamma_initializer='glorot_uniform')(mobilenet.get_layer(name='conv_pad_6_base').output)
    conf1 = layers.Conv2D(4*4*num_classes, kernel_size=3, padding='same')(x)
    conf1 = layers.Reshape((conf1.shape[1]*conf1.shape[2]*conf1.shape[3]//num_classes, num_classes))(conf1)
    loc1 = layers.Conv2D(4*4*4, kernel_size=3, padding='same')(x)
    loc1 = layers.Reshape((loc1.shape[1]*loc1.shape[2]*loc1.shape[3]//4, 4))(loc1)

    x = layers.MaxPool2D(3, 1, padding='same')(mobilenet.get_layer(name='conv_pad_12_base').output)
    x = layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu')(x)
    x = layers.Conv2D(1024, 1, padding='same', activation='relu')(x)
    conf2 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)
    conf2 = layers.Reshape((conf2.shape[1] * conf2.shape[2] * conf2.shape[3] // num_classes, num_classes))(conf2)
    loc2 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)
    loc2 = layers.Reshape((loc2.shape[1]*loc2.shape[2]*loc2.shape[3]//4, 4))(loc2)

    x = layers.Conv2D(256, 1, activation='relu')(x)
    x = layers.Conv2D(512, 3, strides=2, padding='same', activation='relu')(x)
    conf3 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)
    conf3 = layers.Reshape((conf3.shape[1] * conf3.shape[2] * conf3.shape[3] // num_classes, num_classes))(conf3)
    loc3 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)
    loc3 = layers.Reshape((loc3.shape[1] * loc3.shape[2] * loc3.shape[3] // 4, 4))(loc3)

    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    conf4 = layers.Conv2D(6 * num_classes, kernel_size=3, padding='same')(x)
    conf4 = layers.Reshape((conf4.shape[1] * conf4.shape[2] * conf4.shape[3] // num_classes, num_classes))(conf4)
    loc4 = layers.Conv2D(6 * 4, kernel_size=3, padding='same')(x)
    loc4 = layers.Reshape((loc4.shape[1] * loc4.shape[2] * loc4.shape[3] // 4, 4))(loc4)

    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf5 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)
    conf5 = layers.Reshape((conf5.shape[1] * conf5.shape[2] * conf5.shape[3] // num_classes, num_classes))(conf5)
    loc5 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)
    loc5 = layers.Reshape((loc5.shape[1] * loc5.shape[2] * loc5.shape[3] // 4, 4))(loc5)

    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(256, 3, activation='relu')(x)
    conf6 = layers.Conv2D(4 * num_classes, kernel_size=3, padding='same')(x)
    conf6 = layers.Reshape((conf6.shape[1] * conf6.shape[2] * conf6.shape[3] // num_classes, num_classes))(conf6)
    loc6 = layers.Conv2D(4 * 4, kernel_size=3, padding='same')(x)
    loc6 = layers.Reshape((loc6.shape[1] * loc6.shape[2] * loc6.shape[3] // 4, 4))(loc6)

    confs = layers.concatenate([conf1, conf2, conf3, conf4, conf5, conf6], axis=1)
    locs = layers.concatenate([loc1, loc2, loc3, loc4, loc5, loc6], axis=1)
    model = tf.keras.Model(inputs=mobilenet.layers[0].output, outputs=[confs, locs])

    return model

if __name__ == '__main__':
    num_classes = 10
    model = get_mobilenet_SSD(image_size=(300, 300, 3), num_classes=num_classes)

    print(model.summary())

    image = np.random.rand(1, 300, 300, 3)
    confs, locs = model.predict(image)
    print('confs shape =', np.shape(confs))
    print('locs shape =', np.shape(locs))
