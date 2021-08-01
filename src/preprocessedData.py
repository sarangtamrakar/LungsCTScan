__Author__ = 'SARANG TAMRAKAR'
__Email__ = 'sarang.tamrakarsgi15@gmail.com'
__GitHub__ = 'sarangtamrakar'
__Version__ = '1.0.0'
try:
    import yaml
    import logging
    import json
    import os
    import nibabel as nib
    import numpy as np
    import cv2
    import imgaug as ia
    import imgaug.augmenters as iaa
    from configInfo import readConfig
except Exception as e:
    raise e


class PreprocessClass:
    """
        class Name: PreprocessClass
        WrittenBy: SARANG TAMRAKAR
        Version: 1.0
        Description: This class specially designed for preprocess the .nii data to Binary format.
    """
    def __init__(self):
        pass

    def read_nii(self,file_path):
        """
                Method Name: dice_coef
                WrittenBy: SARANG TAMRAKAR
                Version: 1.0
                Description: this method it reads the .nii data & return array formated data

                return: It returns the array
        """
        data = nib.load(file_path)
        array = data.get_fdata()
        array = np.rot90(np.array(array))
        return array

    def DataAugmentation(self,images,masks):
        """
                Method Name: DataAugmentation
                WrittenBy: SARANG TAMRAKAR
                Version: 1.0
                Description: this method apply the Data Augmentation on the 2D images & their respective Masks
                Save: It save the data into .npy format.
        """
        try:
            # define lambda function...
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)

            seq = iaa.Sequential([
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-40, 40),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    ))], random_order=True)

            imgs_aug, masks_aug = seq(images=images, segmentation_maps=masks)

            # merging the original images & masks with augmented images & masks
            cts = np.concatenate((images, imgs_aug), axis=0)
            infections = np.concatenate((masks, masks_aug), axis=0)

            # saving it in .npy format
            # np.save('imgs_complete.npy', cts)
            # np.save('masks_complete.npy', infections)
            return cts,infections

        except Exception as e:
            raise e

    def PrepareTrainingData(self):
        """
                        Method Name: DataAugmentation
                        WrittenBy: SARANG TAMRAKAR
                        Version: 1.0
                        Description: In this method we will prepare the data from .nii format to array
                        return: It returns the dice_coef of actual & predicted values.
        """
        try:
            config_file = readConfig()
            datadir = config_file['datapreprocesseing']['Rawdata_dir']
            image_dir = config_file['datapreprocesseing']['trainImgPath']
            mask_dir = config_file['datapreprocesseing']['trainMaskPath']
            im_height = config_file['datapreprocesseing']['im_height']
            im_width = config_file['datapreprocesseing']['im_width']
            completeimages = config_file['datapreprocesseing']['completeimages']
            completemasks = config_file['datapreprocesseing']['completemasks']


            # print(datadir)
            # print(image_dir)
            # print(mask_dir)
            # print(im_height)
            # print(im_width)

            # getting the list of training images & training mask in Train dir...
            train_img_list = os.listdir(image_dir)
            train_mask_list = os.listdir(mask_dir)
            # creating blank list for storing all 2D images & masks
            training_images = []
            training_masks = []

            for mask, img in zip(train_mask_list, train_img_list):  # itereting through each image & mask
                training_image = self.read_nii(str(image_dir) + str(img))
                training_mask = self.read_nii(str(mask_dir) + str(mask))

                # getting no of slices along the z-axis..
                slices = training_image.shape[2]

                # we are trying to do axial cut along the Z -axis which is slices
                for k in range(slices):
                    img_2D = np.array(training_image[:, :, k])
                    mask_2D = np.array(training_mask[:, :, k])

                    # resize them so that we can store in array
                    img_2D = cv2.resize(img_2D, (im_height, im_width))
                    mask_2D = cv2.resize(mask_2D, (im_height, im_width))

                    if len(np.unique(mask_2D)) != 1:
                        training_images.append(img_2D)
                        training_masks.append(mask_2D)


            # convert list to array
            training_images = np.array(training_images)
            training_masks = np.array(training_masks)

            # convert the data type to uint8 because augmentation takes only that data type
            training_images = np.uint8(training_images)
            training_masks = np.uint8(training_masks)

            # Apply Data augmentation on images & their respective masks
            cts,infections = self.DataAugmentation(training_images,training_masks)

            # saving train imgs & mask in .npy format to pass the model.
            np.save(datadir+completeimages,cts)
            np.save(datadir +completemasks,infections)
            print("Augmented Training imgs & masks saved in .npy format")

        except Exception as e:
            raise e



