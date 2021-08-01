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
    import pandas as pd
    from src.PerformanceMetrics import PerformanceClass
    from src.preprocessedData import PreprocessClass
    from src.splitData import SplitDataClass
    from src.UnetModel import ModelBuildClass
    from configInfo import readConfig
    from tensorflow.keras import layers, metrics, models, losses, callbacks
except Exception as e:
    raise e



class TrainClass:
    """
            class Name: TrainClass
            WrittenBy: SARANG TAMRAKAR
            Version: 1.0
            Description: This class specially designed for Training the UNet model on Augmentated data.
    """
    def __init__(self):
        configData = readConfig()
        self.im_height = configData['ModelBuilding']['im_height']
        self.im_width = configData['ModelBuilding']['im_width']
        self.ModelCheckPointDir = configData['ModelBuilding']['ModelCheckPointDir']
        self.TensorBoardLogsDir = configData['ModelBuilding']['TensorBoardLogsDir']
        self.Epochs = configData['ModelBuilding']['Epochs']
        self.batchSize = configData['ModelBuilding']['batchSize']
        self.historyCSV = configData['ModelBuilding']['historyDf']
        self.PerformanceClass = PerformanceClass()
        self.PreprocessClass = PreprocessClass()
        self.SplitDataClass = SplitDataClass()
        self.ModelBuildClass =  ModelBuildClass()

    def TrainModel(self):
        """
                Method Name: TrainModel
                WrittenBy: SARANG TAMRAKAR
                Version: 1.0
                Description: This method Train the Unet model & save the logs in dirs
                return: None
        """
        try:
            # Load & split the completeData
            xtrain,ytrain,xtest,ytest = self.SplitDataClass.GetsplitedData()

            # calling the model
            unetModel = self.ModelBuildClass.getUnet()

            # adding the ModelCheckPointing & TensorBoardLogs
            checkp = callbacks.ModelCheckpoint(self.ModelCheckPointDir,save_best_only=True,monitor="val_loss",mode='min')
            board = callbacks.TensorBoard(self.TensorBoardLogsDir)

            # fitting the data to model
            history = unetModel.fit(x=xtrain, y=ytrain, batch_size=self.batchSize, callbacks=[checkp, board], epochs=self.Epochs,
                            validation_data=(xtest, ytest))

            # saving the history in csv format
            historyDF = pd.DataFrame(history.history)
            historyDF.to_csv(self.historyCSV,header=True)

        except Exception as e:
            raise e






