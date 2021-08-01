__Author__ = 'SARANG TAMRAKAR'
__Email__ = 'sarang.tamrakarsgi15@gmail.com'
__GitHub__ = 'sarangtamrakar'
__Version__ = '1.0.0'

try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    import numpy as np
    from configInfo import readConfig
    from src.preprocessedData import PreprocessClass
except Exception as e:
    raise e


class SplitDataClass:
    """
            class Name: SplitDataClass
            WrittenBy: SARANG TAMRAKAR
            Version: 1.0
            Description: This class specially designed for Loading the prepocessed Data & splitting purpose.
    """
    def __init__(self):
        configData = readConfig()
        self.testSize = configData['splitData']['testSize']
        self.shuffle = configData['splitData']['shuffle']
        self.completeImages = configData['datapreprocesseing']['completeimages']
        self.completeMasks = configData['datapreprocesseing']['completemasks']

    def readCompleteData(self):
        """
                        Method Name: readCompleteData
                        WrittenBy: SARANG TAMRAKAR
                        Version: 1.0
                        Description: This method Load the Preprocessed data from .npy format.
                        return: It returns training_images,training_masks

        """
        try:
            completeImages = np.load(self.completeImages)
            completeMasks = np.load(self.completeMasks)
            return completeImages, completeMasks
        except Exception as e:
            raise e

    def GetsplitedData(self):
        """
                                Method Name: GetsplitedData
                                WrittenBy: SARANG TAMRAKAR
                                Version: 1.0
                                Description: This method split the data into train test.
                                return: It returns  xtrain,ytrain,xtest,ytest

        """
        try:
            completeImages, completeMasks = self.readCompleteData()

            # let's split the dataset
            xtrain, xtest, ytrain, ytest = train_test_split(completeImages, completeMasks, test_size=self.testSize,
                                                            shuffle=self.shuffle)

            # expand the dimension to 4D tensors

            xtrain = tf.expand_dims(xtrain, axis=3)
            ytrain = tf.expand_dims(ytrain, axis=3)

            xtest = tf.expand_dims(xtest, axis=3)
            ytest = tf.expand_dims(ytest, axis=3)

            return xtrain, ytrain, xtest, ytest
        except Exception as e:
            raise e


