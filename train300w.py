import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tqdm.notebook import tqdm
import tensorflow as tf
############################################
from SelfattentionModel import *
##############################################
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping,LearningRateScheduler
from sklearn.utils import shuffle
from tensorflow.keras.models import *
from CustomCallback import AdditionalValidationSets
from  config import Constant
from learning_rate_schedulers import StepDecay
from learning_rate_schedulers import PolynomialDecay
#######################################################
from tensorflow.keras.applications.resnet import preprocess_input
#######################################################
def train(args):
    output_folder=args.save_path
    image_file = args.train_image_path
    keypoint_file = args.train_keypoint_path
    eval_image_full = args.eval_images_full
    eval_keypoint_full = args.eval_keypoints_full
    eval_image_comm = args.eval_images_comm
    eval_keypoint_comm = args.eval_keypoints_comm
    eval_image_ch = args.eval_images_ch
    eval_keypoint_ch = args.eval_keypoints_ch
    epochs = Constant.epoch
    schedule = Constant.schedule
    pretrained_file = Constant.pretrained_file
    checkpoint_file = output_folder+'/checkpoint-{epoch:02d}.h5'
    model_file = output_folder+"/MyModel.hd5"  
    hist_csv_file = output_folder+'/MyModelhistory.csv'  
    fig_file = output_folder+"/MyModel_Lose.jpg"    
    ########################################
    train_images = np.load(image_file)
    train_keypoints = np.load(keypoint_file)
    train_images=train_images
    train_keypoints=train_keypoints
    validate_images_full = np.load(eval_image_full)
    validate_images_full = validate_images_full
    validate_keypoints_full = np.load(eval_keypoint_full)
    validate_keypoints_full = validate_keypoints_full
    validate_images_comm = np.load(eval_image_comm)
    validate_images_comm = validate_images_comm
    validate_keypoints_comm = np.load(eval_keypoint_comm)
    validate_keypoints_comm = validate_keypoints_comm
    validate_images_ch = np.load(eval_image_ch)
    validate_images_ch = validate_images_ch
    validate_keypoints_ch = np.load(eval_keypoint_ch)
    validate_keypoints_ch = validate_keypoints_ch
    print(np.shape(train_images))
    print(np.shape(train_keypoints))
    print(np.shape(validate_images_full)))

    ########################################
    if Constant.schedule == "step":
        print("[INFO] using 'step-based' learning rate decay...")
        schedule = StepDecay(initAlpha=1e-4, factor=0.1, dropEvery=100)
    elif Constant.schedule == "linear":
        print("[INFO] using 'linear' learning rate decay...")
        schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=1)
    elif Constant.schedule == "poly":
        print("[INFO] using 'polynomial' learning rate decay...")
        schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=1e-1, power=5)
    
    ########################################
    if(pretrained_file):
        model = build_model(input_shape=(128,128,3),pretrained_file=pretrained_file)
    else:
        model = build_model(input_shape=(128,128,3))#pretrained_file=pretrained_file)
    history_callback = AdditionalValidationSets([(validate_images_full, validate_keypoints_full, '300W_Full_eval'),(validate_images_comm, validate_keypoints_comm, '300W_comm_eval'),(validate_images_ch, validate_keypoints_ch, '300W_ch_eval')])
    callbacks=[ModelCheckpoint(checkpoint_file),LearningRateScheduler(schedule),history_callback]
    #######################################################################3
    #Train the model
    history = model.fit(train_images,[train_keypoints,train_keypoints,train_keypoints],validation_data=(validate_images_full,[validate_keypoints_full,validate_keypoints_full,validate_keypoints_full]),batch_size = Constant.batch_size,epochs = epochs,callbacks=callbacks,verbose=2,initial_epoch=Constant.initial_epoch)
    model.save(model_file)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(fig_file)
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    with open(hist_csv_file, mode='a') as f:
        hist_df.to_csv(f)    


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prepare")
    parser.add_argument('--train_image_path',dest='train_image_path',help='path to .npy file of training images')
    parser.add_argument('--train_keypoint_path',dest='train_keypoint_path',help='path to .npy file of training keypoint')
    parser.add_argument('--eval_images_full',dest='eval_images_full',help='path to .npy file of testing images')
    parser.add_argument('--eval_keypoints_full',dest='eval_keypoints_full',help='path to .npy file of testing keypoint')
    parser.add_argument('--eval_images_comm',dest='eval_images_comm',help='path to .npy file of testing images')
    parser.add_argument('--eval_keypoints_comm',dest='eval_keypoints_comm',help='path to .npy file of testing keypoint')  
    parser.add_argument('--eval_images_ch',dest='eval_images_ch',help='path to .npy file of testing images')
    parser.add_argument('--eval_keypoints_ch',dest='eval_keypoints_ch',help='path to .npy file of testing keypoint')    
    parser.add_argument('--save_path',dest='save_path',help='path to output folder')
    args = parser.parse_args()
    train(args)