import os
import shutil

import tensorflow as tf
from tensorflow.keras.models import load_model
tf.keras.backend.set_image_data_format('channels_last')
from tensorflow.keras import backend
import cv2
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from configInfo import readConfig



class PreditionClass:
    """
            class Name: PreditionClass
            WrittenBy: SARANG TAMRAKAR
            Version: 1.0
            Description: This class specially designed for prediction on the .nii data.
        """
    def __init__(self):
        self.configData = readConfig()
        self.modelname = self.configData['Predition']['ModelPath']
        self.preditionFileName = self.configData['Predition']['PredictionFilePath']
        self.Model = load_model(self.modelname, custom_objects={"dice_coef": self.dice_coef})

    def dice_coef(self,y_true, y_pred):
        smooth = 1
        y_true_f = backend.flatten(y_true)  # converting the complete dim into 1D array
        y_pred_f = backend.flatten(y_pred)  # for match ypred & ytrue
        intersection = backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

    def read_nii(self,file_path):
        """
                Method Name: read_nii
                WrittenBy: SARANG TAMRAKAR
                Version: 1.0
                Description: this method it reads the .nii data & return array formated data

                return: It returns the array
        """
        data = nib.load(file_path)
        array = data.get_fdata()
        array = np.rot90(np.array(array))
        return array

    def Prediction(self,input_start,input_end):
        """
                        Method Name: Prediction
                        WrittenBy: SARANG TAMRAKAR
                        Version: 1.0
                        Description: this method do the prediction on input medical image

                        return: It returns the array
                """
        try:
            path = 'static/Records/'
            try:
                for im in os.listdir(path):
                    try:
                        os.remove(im)
                    except:
                        continue
            except:
                pass


            images = self.read_nii(self.preditionFileName)
            slices = images.shape[2]
            if (input_start > slices) or (input_end > slices):
                return "Please give slices range under {} range".format(slices)

            train_images = []
            for slic in range(input_start,input_end):
                img = np.array(images[:, :, slic])
                img2D = cv2.resize(img, (256, 256))
                train_images.append(img2D)

            train_images = np.array(train_images).astype("uint8")
            train_images = tf.expand_dims(train_images, axis=3)
            result = self.Model.predict(train_images)
            result = result.squeeze(axis=3)
            for i in range(len(result)):
                img = result[i]
                plt.imsave("static/Records/{}.png".format(i), img)



            """
            new_images = []
            mid = len(train_images)//2
            start = mid - 2
            end = mid + 3
            for idx in range(start,end):
                new_images.append(train_images[idx])

            train_images = new_images


            train_images = np.array(train_images).astype("uint8")
            train_images = tf.expand_dims(train_images, axis=3)
            result = self.Model.predict(train_images)
            result = result.squeeze(axis=3)

            
            mid = len(result) // 2
            start = mid - 2
            end = mid + 3
            

            for i,j in zip(range(len(result)),range(148,153)):
                img = result[i]
                plt.imsave("static/Records/{}.png".format(j), img)
            """
            return True

        except Exception as e:
            print("Exception : "+str(e))
            return False
