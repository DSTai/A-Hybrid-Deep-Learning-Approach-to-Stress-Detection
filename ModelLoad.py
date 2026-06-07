import os
import sys
from keras.models import load_model
import pandas as pd
import glob
import argparse
# from sklearn import preprocessing
import numpy as np
from scipy import stats
# from tensorflow.python.keras.utils.np_utils import to_categorical

RESAMPLE = 175
TIME_PERIODS = 40
STEP_DISTANCE = 40
LABELS = [1, 2, 3]
N_FEATURES = 1
# num_classes = len(LABELS) - 1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_data(filepath):
        print(f"Reading data from: {filepath}")
        # print(filepath)
        os.chdir(filepath)
        extension = 'pkl'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        # print(all_filenames)
        frames = []
        label = []
        RESP = []
        for f in all_filenames:
            df = pd.read_pickle(f)
            # print(df)
            list_label = df['label'].tolist()
            list_chest_RESP = df['signal']['chest']['Resp'].tolist()

            # print('length of label: ', len(list_label))
            for i in range(0, int(len(list_chest_RESP)/RESAMPLE)):
                RESP.append(list_chest_RESP[i*RESAMPLE][0])
            # print(int(len(list_label)/RESAMPLE))
            count_label = [0, 0]
            for j in range(0, int(len(list_label)/RESAMPLE)):
                if list_label[j*RESAMPLE] == 2:
                    count_label[1] = count_label[1] + 1
                else:
                    count_label[0] = count_label[0] + 1

                label.append(list_label[j*RESAMPLE])
            # print(count_label)
            # print('length of RESP: ', len(RESP))
            # print('length of label: ', len(label))
            df_fn = pd.DataFrame(list(zip(label,RESP)),columns=['label','RESP'])
            frames.append(df_fn)
        df_frames = pd.concat(frames)
        # print('finished loading data...')
        return df_frames

def create_segments_and_labels(df, time_steps, step):
    segments = []
    labels = []

    for i in range(0, len(df) - time_steps, step):
        resp = df['RESP'].values[i: i + time_steps]
        label = stats.mode(df['label'][i: i + time_steps], keepdims=True)[0][0]
        #label = stats.mode(df['label'][i: i + time_steps])[0]

        if label in LABELS:
            segments.append([resp])
            if (label == 1) or (label == 3):
                labels.append("non-stress")
            else:
                labels.append("stress")

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    # return reshaped_segments, labels
    return reshaped_segments

def dataPreprocess():
    # argument_parser = argparse.ArgumentParser()
    # argument_parser.add_argument("--data", type=str, help="Data file (.pkl) is required")
    # args = argument_parser.parse_args()

    # le = preprocessing.LabelEncoder()

    user_data = read_data(os.path.join(BASE_DIR, "selectedData"))

    # reshaped_segments_user_data, labels = create_segments_and_labels(user_data, TIME_PERIODS, STEP_DISTANCE)
    reshaped_segments_user_data = create_segments_and_labels(user_data, TIME_PERIODS, STEP_DISTANCE)

    num_time_periods, num_sensors = reshaped_segments_user_data.shape[1], reshaped_segments_user_data.shape[2]
    
    i_shape = (num_time_periods*num_sensors)
    
    reshaped_segments_user_data = reshaped_segments_user_data.reshape(reshaped_segments_user_data.shape[0], i_shape)

    reshaped_segments_user_data = reshaped_segments_user_data.astype('float32')

    reshaped_segments_user_data = np.asarray(reshaped_segments_user_data)

    # labels = le.fit_transform(labels)

    # labels = to_categorical(labels, num_classes)

    # return reshaped_segments_user_data, labels
    return reshaped_segments_user_data

def modelLoad(username):
    user_number = username[4:]
    #model_imb_al_reinforce_path = r"D:\Documents\Downloads\THESIS\PythonServer\model\best_model_s13.hdf5"
    model_imb_al_reinforce_path = os.path.join(
    BASE_DIR,
    "model",
    f"best_model_s{user_number}.hdf5"
)
    print(f"Loading model from {model_imb_al_reinforce_path}...")
    model_imb_al_reinforce = load_model(model_imb_al_reinforce_path)
    print("Model loaded successfully!")
    # reshaped_segments_user_data, labels = dataPreprocess()
    reshaped_segments_user_data = dataPreprocess()

    predictions = model_imb_al_reinforce.predict(reshaped_segments_user_data)
    
    # classification labels (output) predicted from input data
    max_predictions = np.argmax(predictions, axis=1) 
    
    # true labels from dataset. These labels represent the actual or correct outcomes 
    # -> used for training or testing process, need to demonstrate the accuracy of the model
    # max_train (true labels) and max_predictions (predicted labels) used to calculate how well your model is performing
    # max_train = np.argmax(labels, axis=1) 
    
    # print(predictions)
    # print(max_predictions)
    # print(max_train)

    return reshaped_segments_user_data, max_predictions, predictions