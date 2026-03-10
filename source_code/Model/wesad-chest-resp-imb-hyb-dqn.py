#import stumpy
import os
os.environ['PYTHONHASHSEED']= '0'
import numpy as np
np.random.seed(1)
import random as rn
rn.seed(1)
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import callbacks
#from tensorflow.keras import optimizers
#%matplotlib inline
import pandas as pd
import argparse
#import seaborn as sns
import csv
from numpy import save,load
import time
#import coremltools
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import preprocessing as pre
from scipy.stats import dirichlet
from matplotlib import pyplot as plt
import math
import sys
import glob
#from keras.layers import *
#from keras.models import *
#import keras.backend as K
#from keras.callbacks import ModelCheckpoint
#from layers import AttentionWithContext, Addition
#from collections import deque
#import random
#from fastdtw import fastdtw
#from scipy.spatial.distance import euclidean
#import mass_ts as mts


#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Reshape, LSTM
#from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint
#from keras import optimizers


import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#=========================================#
#                                         #
#             BEGIN                       #
#                                         #
#=========================================#
# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
#sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
#print('keras version ', keras.__version__)
# Same labels will be reused throughout the program

LABELS = [1, 2, 3]
#LABELS = ["Desk Work","Eating/Drinking","Movement","Sport","unknown"]
# The number of steps within one time segment
TIME_PERIODS = 40
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40
N_FEATURES = 1
RESAMPLE = 175
BATCH_SIZE = 64
EPOCHS = 8
EPISODES = 5
subject = 6

from datetime import datetime
now = datetime.now()
dtime = now.strftime('%H-%M-%S-%d-%m-%Y')
print('date and time: ', dtime)


path = "/results/wesad/test-S" + str(subject) + "/10s/chest/RESP/bias_imb_hyb_dqn/reward-2labels/epochs" + str(EPOCHS) +"/hyb_dqn" + "eps_" + \
       str(EPISODES) + "_epoch_" + str(EPOCHS) + "-2_" + dtime + ".txt"
cwd = os.getcwd()
report_dir = cwd + path
# Automatically create directories if they don't exist
os.makedirs(os.path.dirname(report_dir), exist_ok=True)

WEIGHT_CLASS = []
num_classes = len(LABELS) - 1
print(num_classes)

def write_list_to_file(guest_list, filename):
    """Write the list to csv file."""
    with open(filename, "w") as output:
        writer = csv.writer(output, delimiter = ',', lineterminator='\n')
        for row in enumerate(guest_list):
            writer.writerows([row])

def read_data(cwd, filepath):
    #print('====loading data====')
    #df = pd.concat(map(pd.read_pickle, glob.glob(os.path.join('', filepath))))
    # Parse paths
    print(filepath)
    os.chdir(cwd + "/" + filepath)
    extension = 'pkl'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    # combine all files in the list
    frames = []
    label = []
    ECG = []
    EDA = []
    BVP = []
    EMG = []
    RESP = []
    TEMP = []
    for f in all_filenames:
        df = pd.read_pickle(f)
        print(df)
        #np.set_printoptions(threshold=sys.maxsize)
        list_label = df['label'].tolist()
        #print('labels = ', list_label)
        #list_chest_ECG = df['signal']['chest']['ECG'].tolist()
        #list_chest_EDA = df['signal']['chest']['EDA'].tolist()
        #list_chest_EMG = df['signal']['chest']['EMG'].tolist()
        list_chest_RESP = df['signal']['chest']['Resp'].tolist()
        #list_chest_TEMP = df['signal']['chest']['Temp'].tolist()
        #list_wrist_EDA = df['signal']['wrist']['EDA'].tolist()
        #list_wrist_BVP = df['signal']['wrist']['BVP'].tolist()
        #print(f)
        #print('length of BVP: ', len(list_wrist_BVP))
        #print('length of EDA: ', len(list_wrist_EDA))
        #print('length of EDA: ', len(list_wrist_EDA))

        print('length of label: ', len(list_label))
        for i in range(0, int(len(list_chest_RESP)/RESAMPLE)):
            #ECG.append(list_chest_ECG[i*RESAMPLE][0])
            RESP.append(list_chest_RESP[i*RESAMPLE][0])
            #EDA.append(list_chest_EDA[i*RESAMPLE][0])
            #EMG.append(list_chest_EMG[i*RESAMPLE][0])
            #TEMP.append(list_chest_TEMP[i*RESAMPLE][0])
            #EDA.append(list_wrist_EDA[i][0])
            #BVP.append(list_wrist_BVP[i][0])
        print(int(len(list_label)/RESAMPLE))
        #print(list_label)
        count_label = [0, 0]
        for j in range(0, int(len(list_label)/RESAMPLE)):
            if list_label[j*RESAMPLE] == 2:
                count_label[1] = count_label[1] + 1
            else:
                count_label[0] = count_label[0] + 1

            label.append(list_label[j*RESAMPLE])
        print(count_label)
        print('length of RESP: ', len(RESP))
        print('length of label: ', len(label))
        #plt.plot(ECG)
        #plt.show()
        df_fn = pd.DataFrame(list(zip(label,
                                      #ECG,
                                      RESP,
                                      #EDA,
                                      #BVP
                                      #EMG,
                                      #TEMP
                                      )),
                             columns=['label',
                                      #'ECG',
                                      'RESP',
                                      #'EDA',
                                      #'BVP'
                                      #'EMG',
                                      #'TEMP'
                                      ])
        frames.append(df_fn)
    df_frames = pd.concat(frames)

    #df = pd.read_pickle(filepath)
    #
    #print(df['label'])
    #print(df['signal']['chest']['ECG'])
    #print(df['signal']['chest']['EDA'])
    #print(df['signal']['chest']['EMG'])


    #print(len(df['signal']['chest']['RESP']))
    #print(len(df['signal']['chest']['TEMP']))
    #print(df['label'])

    #df = pd.read_csv(filepath, skip_blank_lines=True, na_filter=True).dropna()
    #df = df.drop(labels=['timestamp'], axis=1)
    #df.dropna(how='any', inplace=True)
    #cols_to_norm = ['mean_x_p', 'mean_y_p', 'mean_z_p', 'var_x_p', 'var_y_p', 'var_z_p',
    #                'mean_x_w', 'mean_y_w', 'mean_z_w', 'var_x_w', 'var_y_w', 'var_z_w']
    #df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    # Round numbers
    #df = df.round({'attr_x': 4, 'attr_y': 4, 'attr_z': 4})
    print('finished loading data...')
    return df_frames

def create_segments_and_labels(df, time_steps, step):
    print('=====starting to segment====')
    # x, y, z acceleration as features

    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    count_label = [0, 0]
    for i in range(0, len(df) - time_steps, step):
        #x = df['ECG'].values[i: i + time_steps]
        #eda = df['EDA'].values[i: i + time_steps]
        resp = df['RESP'].values[i: i + time_steps]
        #bvp = df['BVP'].values[i: i + time_steps]
        #z = df['EMG'].values[i: i + time_steps]
        #t = df['Temp'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        #print(df['label'][i: i + time_steps])[0][0]
        #print(df['label'][i: i + time_steps])[0][i + time_steps]
        label = stats.mode(df['label'][i: i + time_steps])[0][0]
        if label in LABELS:
            segments.append([#x,
                             #eda,
                             resp,
                             #bvp
                             #z,
                             #t
                            ])
            if (label == 1) or (label == 3):
                labels.append("non-stress")
            else:
                labels.append("stress")
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)
    print('=====end segment====')
    return reshaped_segments, labels

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def show_basic_dataframe_info(dataframe):
    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))

# def show_confusion_matrix(validations, predictions):
#
#     matrix = metrics.confusion_matrix(validations, predictions)
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(matrix,
#                 cmap='coolwarm',
#                 linecolor='white',
#                 linewidths=1,
#                 xticklabels=LABELS,
#                 yticklabels=LABELS,
#                 annot=True,
#                 fmt='d')
#     plt.title('Confusion Matrix ACTIVITY')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()

def calculate_non_weighted_entropy(array):
    list_entropy = []
    total_rows = array.shape[1]
    for i in range(total_rows):
        entropy = 0
        total_columns = array.shape[0]
        for j in range(total_columns):
            if (array[j][i] != 0):
                entropy += (array[j][i] * math.log(array[j][i]))
        #print('entropy: ', entropy)
        list_entropy.append(-entropy)

    return list_entropy

def concatenate_delete(train_xs, train_ys, validation_xs, validation_ys, delete_indices):
    train_xs = np.concatenate((train_xs, validation_xs[delete_indices]))
    train_ys = np.concatenate((train_ys, validation_ys[delete_indices]))
    #f_train.extend(np.array(f_validation)[delete_indices].tolist())

    validation_xs = np.delete(validation_xs, delete_indices, 0)
    validation_ys = np.delete(validation_ys, delete_indices, 0)
    #f_validation = np.delete(f_validation, delete_indices).tolist()

    return train_xs, train_ys, validation_xs, validation_ys

def concatenate(train_xs, train_ys, validation_xs, validation_ys, delete_indices):

    train_xs = np.concatenate((train_xs, validation_xs[delete_indices]))
    train_ys = np.concatenate((train_ys, validation_ys[delete_indices]))

    return train_xs, train_ys

def indicesNotMatches(listA, listB):
    # index variable
    idx = 0

    # Result list
    res = []

    # With iteration
    for i in listA:
        if i != listB[idx]:
            res.append(idx)
        idx = idx + 1

    # Result
    print("The index positions with mismatched values:\n", res)
    return res


def remove_common(a, b):
    a = a.tolist()
    #b = b.tolist()

    for i in a[:]:
        if i in b:
            a.remove(i)
            b.remove(i)

    print("list1 : ", a)
    print("list2 : ", b)
    return a

def count_classes_function(array_inputs):
    weight_classes = [0, 0]
    ratio_classes = [0, 0]
    count_classes_train_set = [0, 0]
    highest_class = 0
    count_classes_train_set_all = 0
    for index_, value_ in enumerate(array_inputs):
        if (np.argwhere(value_)) == 0:
            count_classes_train_set[0] = count_classes_train_set[0] + 1
        elif (np.argwhere(value_)) == 1:
            count_classes_train_set[1] = count_classes_train_set[1] + 1

    for ide, v in enumerate(count_classes_train_set):
        # output_file.write("class %s %s \n" % (ide, count_classes_train_set[ide]))
        print("class %s %s \n" % (ide, count_classes_train_set[ide]))
        count_classes_train_set_all = count_classes_train_set_all + count_classes_train_set[ide]
        if count_classes_train_set[ide] > highest_class:
            highest_class = count_classes_train_set[ide]
    print("highest class:", highest_class)
    # output_file.write("total classes: %s  \n" % (count_classes_train_set_all))

    for ide, v in enumerate(count_classes_train_set):
        weight_classes[ide] = (v/ count_classes_train_set_all * 1.0)
        ratio_classes[ide] = int(highest_class / (v))
    # (2) weight_classes[ide] = 1 - (v/(count_classes_train_set_all*1.0))
    # print "weight classes: ", weight_classes

    # ===find the minority class and return the weight of all classes==
    # ======stratgegy (1) weighted with formular true ratio = TP/all_classes ==========
    index_class = np.argsort(weight_classes)[:1]
    smallest_weight = min(weight_classes)
    # print "smallest weight: ", smallest_weight

    # ========stratgegy (2) weighted with formular 1 - true ratio==========
    # index_class = np.argsort(weight_classes)[-1:]
    # largest_weight = max(weight_classes)
    # print "largest weight", largest_weight

    lenght_act = len(weight_classes)
    index_class = lenght_act - index_class - 1
    onehot_minority_class = "%0*d" % (lenght_act, 10 ** index_class)
    onehot_minority_class = [int(l) for l in onehot_minority_class]
    onehot_minority_class = np.hstack([np.expand_dims(x, 0) for x in onehot_minority_class])
    onehot_minority_class = np.asarray(onehot_minority_class)
    print("one hot minority class: ", onehot_minority_class)

    return weight_classes, smallest_weight, onehot_minority_class, ratio_classes

def calculate_weighted_entropy(array, ratio_classes):
    list_entropy = []
    total_rows = array.shape[1]
    # print "total_rows", total_rows

    for i in range(total_rows):
        entropy = 0
        sum_ratio_classes = 0
        total_columns = array.shape[0]
        # print "total_columns",total_columns
        #for l in range(total_columns):
        #	sum_ratio_classes += list_ratio_classes[l] * array[l][i]
        #print('sum ratio:',sum_ratio_classes)
        for j in range(total_columns):
            if (array[j][i] != 0):
                #x = float(array[j][i] * list_ratio_classes[j]) / sum_ratio_classes
                #x = float(array[j][i] * list_ratio_classes[j])
                #entropy += x * math.log(x)
                print('ratio_classes[j]: ', ratio_classes[j])
                entropy += (array[j][i]* math.log(array[j][i]))* ratio_classes[j]
        list_entropy.append(-entropy)

    return list_entropy

class DDQNAgent:
    def __init__(self, state_size, action_size, action):
        self.state_size = state_size
        self.action_size = action_size
        self.action = action
        self.memory = []  # For experience replay
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.001  # Exploration rate
        self.epsilon_min = 0.001  # Minimum value of epsilon
        self.epsilon_decay = 0.9  # Gradually reducing the exploration factor
        self.hidden_units = 128
        self.dense_units = 2
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):

        model_m = keras.Sequential()

        w_c, s_w, min_c, r_c = count_classes_function(self.action)

        print('weighted classes before compute with reward: ', w_c)

        #print('self.reward: ', self.reward)

        #input_shape = TIME_PERIODS * 1

        #model_m.add(layers.Reshape((TIME_PERIODS, N_FEATURES),
        #                           input_shape=(input_shape,))
        #            )

        #model_m.add(layers.LSTM(self.hidden_units,
        #                        activation='tanh',
        #                        input_shape=(TIME_PERIODS, N_FEATURES)
        #                        )
        #            )

        #model_m.add(layers.Dropout(0.5))

        #model_m.add(layers.Dense(self.dense_units,
        #                         use_bias=True,
        #                         bias_initializer=keras.initializers.Constant(w_c),
        #                         #kernel_initializer=keras.initializers.Constant(self.reward[:self.hidden_units]),
        #                         activation='softmax')
        #            )

        #model_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #model_m.summary()

        input_shape = TIME_PERIODS * 1

        model_m.add(layers.Reshape((TIME_PERIODS, N_FEATURES),
                                   input_shape=(input_shape,))
                    )
        # Add 1D Convolutional layers
        model_m.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(TIME_PERIODS, N_FEATURES)))
        model_m.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        model_m.add(layers.MaxPooling1D(pool_size=2))
        model_m.add(layers.Dropout(0.5))

        # Add LSTM layers
        model_m.add(layers.LSTM(self.hidden_units,activation='tanh', return_sequences=True))
        model_m.add(layers.LSTM(self.hidden_units))
        model_m.add(layers.Dropout(0.5))

        # Add a Dense layer with softmax activation
        model_m.add(layers.Dense(self.hidden_units, activation='relu'))
        model_m.add(layers.Dense(self.dense_units, use_bias=True,
                                 bias_initializer=keras.initializers.Constant(w_c),
                                 activation='softmax'))

        model_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model_m.summary()
        return model_m

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #if np.random.rand() <= self.epsilon:
        #    return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        #print('act_values: ', act_values)
        #print('np argmax act_values: ', np.argmax(act_values))
        #return np.argmax(act_values)
        return act_values[0], np.argmax(act_values)

    def replay(self, batch_size):

        #alpha = 0.6

        indices = np.random.choice(len(self.memory), batch_size, replace=False)

        minibatch = [self.memory[idx] for idx in indices]

        #print('minibatch: ', minibatch)

        print('length of minibatch: ', len(minibatch))

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:
                print('self.model.predict(next_state): ', self.model.predict(next_state))

                action_next = np.argmax(self.model.predict(next_state)[0])

                print('action_next: ', action_next)

                # print('self.target_model.predict(next_state)[0][action_next]: ', self.target_model.predict(next_state)[0][action_next])
                print('reward: ', reward)

                print('self.gamma * self.model.predict(next_state)[0][action_next]',
                      self.gamma * self.target_model.predict(state)[0][action_next])

                target += self.gamma * self.target_model.predict(next_state)[0][action_next]

                print('target: ', target)

            target_f = self.model.predict(state)

            print('predicted state: ', target_f)

            print('current action: ', action)

            target_f[0][action] = target

            print('predicted state with reward and gamma: ', target_f[0][action])

            print('final predicted state: ', target_f)

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):

        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):

        self.model.load_weights(name)

    def save(self, name):

        self.model.save_weights(name)


argument_parser = argparse.ArgumentParser(description="CLI for training and testing Sequential Neural Network Model")
argument_parser.add_argument("--train_file", default="\\WESAD\\train", type=str, help="Train file (CSV). Required for training.")
argument_parser.add_argument("--validation_file", type=str, help="Validation file (CSV). Required for validation.")
argument_parser.add_argument("--test_file", default="\\WESAD\\test", type=str, help="Test file (CSV). Required for testing.")
args = argument_parser.parse_args()

# Load data set containing all the data from csv
# Define column name of the label vector
#LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = pre.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
start_time = time.time()

train = read_data(cwd, args.train_file)

x_train, y_train = create_segments_and_labels(train, TIME_PERIODS, STEP_DISTANCE)

#Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
#Set input_shape / reshape for Keras
i_shape = (num_time_periods * num_sensors)

print('input_shape: ', i_shape)

x_train = x_train.reshape(x_train.shape[0], i_shape)

print('x_train shape:', x_train.shape)

x_train = x_train.astype('float64')

#x_train = x_train.reshape(x_train.shape + (1,))

x_train = np.asarray(x_train)

print('x_train: ', x_train)

print('x_train shape: ', x_train.shape)

y_train = le.fit_transform(y_train)

y_train = to_categorical(y_train, num_classes)

print('y_train length: ', len(y_train))

#=================Read Test Data===========================#
test = read_data(cwd, args.test_file)

x_test, y_test = create_segments_and_labels(test, TIME_PERIODS, STEP_DISTANCE)

x_test = x_test.reshape(x_test.shape[0], i_shape)

x_test = x_test.astype('float64')

#x_test = x_test.reshape(x_test.shape + (1,))

x_test = np.asarray(x_test)

print('x_test: ', x_test)

y_test = le.fit_transform(y_test)

y_test = to_categorical(y_test, num_classes)

#============Training Phase=============#
state_size = x_train[0].shape[0]

print('state_size: ', state_size)

action_size = y_train.shape[1]

print('action_size: ', action_size)

action = y_train

agent = DDQNAgent(state_size, action_size, action)
# Create the model with attention, train and evaluate
#model_m = agent.build_model()

# Hyper-parameters
# serialize model to JSON
#filepath="weights_dqn" + str(TIME_PERIODS) + str(STEP_DISTANCE) + str(EPOCHS) + str(BATCH_SIZE) + str(AL) + "_activity.best.hdf5"
filepath = "weights_reward_2labels_dqn" + str(EPISODES) + "_best.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

total_rewards = np.zeros(EPISODES)

step = 0

pre_acc = 0.0

update_target_freq = 2

with open(report_dir, "a") as text_file:

    explored_indices = []

    for e in range(EPISODES):

        X_train = x_train

        Y_train = y_train

        text_file.write('len(x_train): ' + str(len(X_train)) + "\n")

        text_file.write('len(y_train): ' + str(len(Y_train)) + "\n")

        episode_reward = []  # record episode reward

        total_sample_relabel = 0

        reward = 0

        batch_indices = []

        print("Starting to reinforcement learning iteration: ", e)

        for i in range(len(X_train) - 1):

            state = X_train[i].reshape([1, state_size])

            #print('state: ', state)

            action_prob, action = agent.act(state)  # return predicted labels (action) and posterior probability (prob) on training data

            #print('action_prob: ', action_prob)

            action = np.argmax(action_prob)

            print('action: ', action)

            next_state = X_train[i+1].reshape([1, state_size])

            print('Y_train[i]: ', np.argmax(Y_train[i]))

            if action != np.argmax(Y_train[i]):

                reward = np.argmax(Y_train[i])

                done = False

                batch_indices.append(i)

            else:

                reward = np.argmax(Y_train[i])

                done = True

            #episode_reward.append(reward)

            agent.remember(state, action, reward, next_state, done)

        text_file.write('length of batch_indices: ' + str(len(batch_indices)) + "\n")

        agent.model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list, verbose=1)

        print("================Test phase for reinforcement learning==================")

        text_file.write("================Test phase for reinforcement learning==================:" + str(e) + "\n")

        #model_m.load_weights("weights_dqn_stress.best.hdf5")

        print("Loaded model from disk")

        y_pred_test = agent.model.predict(x_test)

        max_y_test = np.argmax(y_test, axis=1)

        max_y_pred_test = np.argmax(y_pred_test, axis=1)

        print(classification_report(max_y_test, max_y_pred_test))

        report = classification_report(max_y_test, max_y_pred_test, digits=2, output_dict=False)

        text_file.write(report + "\n")

        print("=======update the agent action and model===========")

        if len(agent.memory) > BATCH_SIZE:

            agent.replay(BATCH_SIZE)

        if e % update_target_freq == 0:

            agent.update_target_model()

#======Print confusion matrix for training data========
y_pred_train = agent.model.predict(x_train)
#Take the class with the highest probability from the predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)

max_train = np.argmax(y_train, axis=1)

print(classification_report(max_train, max_y_pred_train))

# later...
#=====load weights into new model====
print(cwd)
agent.model.load_weights("weights_reward_2labels_dqn" + str(EPISODES) + "_best.hdf5")

print("Loaded model from disk")

agent.model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

score = agent.model.evaluate(x_test, y_test, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])

print('\nLoss on test data: %0.2f' % score[0])

y_pred_test = agent.model.predict(x_test)

max_y_test = np.argmax(y_test, axis=1)

max_y_pred_test = np.argmax(y_pred_test, axis=1)

print(classification_report(max_y_test, max_y_pred_test))
#=======================================================================================================
end_time = time.time()
total_time_in_seconds = end_time - start_time
print("Completion time took %.2f seconds" % total_time_in_seconds)
