__Author__ = 'SARANG TAMRAKAR'
__Email__ = 'sarang.tamrakarsgi15@gmail.com'
__GitHub__ = 'sarangtamrakar'
__Version__ = '1.0.0'


try:
    from configInfo import readConfig
    import tensorflow as tf
    from tensorflow.keras import layers, metrics, models, losses, callbacks
    from src.PerformanceMetrics import PerformanceClass
except Exception as e:
    raise e


class ModelBuildClass:
    """
        class Name: TrainClass
        WrittenBy: SARANG TAMRAKAR
        Version: 1.0
        Description: This class specially designed for Building the Unet Model from scrach.

    """
    def __init__(self):
        configData = readConfig()
        self.im_height = configData['ModelBuilding']['im_height']
        self.im_width = configData['ModelBuilding']['im_width']
        self.Performance = PerformanceClass()


    def getUnet(self):
        """
            Method Name: getUnet
            WrittenBy: SARANG TAMRAKAR
            Version: 1.0
            Description: this method build the Unet Model by using Tensorflow 2.5.0
                                  & keras 2.5.0 APIs

            return: It returns the compiled Model
        """
        try:
            inputs = layers.Input((self.im_height,self.im_width,1))
            conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
            conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
            conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
            pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
            conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
            pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
            conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

            up6 = layers.concatenate(
                [layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
            conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
            conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

            up7 = layers.concatenate(
                [layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
            conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
            conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

            up8 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2],
                                     axis=3)
            conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
            conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

            up9 = layers.concatenate([layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1],
                                     axis=3)
            conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
            conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

            conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

            model = models.Model(inputs=[inputs], outputs=[conv10])

            # parameters before compile
            model.compile(optimizer='adam', loss=[tf.keras.losses.BinaryCrossentropy()],
                          metrics=["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), self.Performance.dice_coef])

            return model
        except Exception as e:
            raise e
