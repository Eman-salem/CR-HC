from tensorflow import keras
import numpy
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from CustomLoss import *
from  config import Constant
from sklearn.metrics import auc

class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets):
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [3]:
                raise ValueError()
        self.epoch = []
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        csv_file = Constant.Eval_Csv_File 
        f = open(csv_file, "a")
        f.write(str(epoch))
        f.write(",")
        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_Keypoint_ground, validation_set_name = validation_set
            else:
                raise ValueError()

            x,y,validation_Keypoint_predict= self.model.predict(validation_data,batch_size=Constant.batch_size)
            NME,error_per_image = normalized_mean_error(validation_Keypoint_ground,validation_Keypoint_predict)
            NME = NME*100
            ############################################
            # calculate the auc for 0.1
            max_threshold = 0.1
            threshold = np.linspace(0, max_threshold, num=2000)
            accuracys = np.zeros(threshold.shape)
            for i in range(threshold.size):
                accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
            area_under_curve_1 = auc(threshold, accuracys) / max_threshold
            Failur_1 =(np.sum(error_per_image>0.1)* 100. / error_per_image.size)
            ###########################################
            valuename = validation_set_name + '_NME'
            self.history.setdefault(valuename, []).append(NME)
            print(valuename," = ",NME,"\n")
            f.write(str(NME))
            f.write(",")
            f.write(str(area_under_curve_1))
            # f.write(",")
            # f.write(str(Failur_1))
            f.write(",")
        f.write('\n')
        f.close()