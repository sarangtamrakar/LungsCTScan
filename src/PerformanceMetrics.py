__Author__ = 'SARANG TAMRAKAR'
__Email__ = 'sarang.tamrakarsgi15@gmail.com'
__GitHub__ = 'sarangtamrakar'
__Version__ = '1.0.0'


try:
    import tensorflow as tf
    from tensorflow import keras
    # setting images channels to last
    tf.keras.backend.set_image_data_format('channels_last')
    from tensorflow.keras import backend
except Exception as e:
    raise e


class PerformanceClass:
    """
    class Name: PerformanceClass
    WrittenBy: SARANG TAMRAKAR
    Version: 1.0
    Description: This class specially designed for getting custom performance Matrices
    """
    def __init__(self):
        pass

    def dice_coef(self,ytrue,ypred):
        """
                Method Name: dice_coef
                WrittenBy: SARANG TAMRAKAR
                Version: 1.0
                Description: this method defines the custom performance matrices Dice Coefficient
                which is generally used in sementic segmentation.

                return: It returns the dice_coef of actual & predicted values.
        """
        smooth = 1
        y_true_f = backend.flatten(ytrue)  # converting the complete dim into 1D array
        y_pred_f = backend.flatten(ypred)  # for match ypred & ytrue
        intersection = backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

    def dice_coef_loss(self,ytrue,ypred):
        """
                        Method Name: dice_coef_loss
                        WrittenBy: SARANG TAMRAKAR
                        Version: 1.0
                        Description: this method defines the custom performance loss Dice Coefficient loss
                        which is generally used in sementic segmentation.

                        return: It returns the dice_coef loss of actual & predicted values.
        """
        return 1 - self.dice_coef(ytrue,ypred)


