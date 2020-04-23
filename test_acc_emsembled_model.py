# -*- coding: utf-8 -*-
'''
Use all data from UR dataset and Multiple Camera dataset to tune the ensembled model.
'''

import json
import os
import pandas as pd
import csv
from tensorflow import keras
from numpy import array
import matplotlib.pyplot as plt
import sys

# read the video data csv,
# it is faster than running the above code each time

ur_path = '/content/drive/My Drive/636project/ur_data.csv'
mc_fall_path = '/content/drive/My Drive/636project/mc_fall_data.csv'
mc_notfall_path = '/content/drive/My Drive/636project/mc_notfall_data.csv'

def read_data(path):
  video_data = []
  with open(path, mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
      video_data.append(row)
  return video_data

ur_data = read_data(ur_path)
mc_fall_data = read_data(mc_fall_path)
mc_notfall_data = read_data(mc_notfall_path)

print(ur_data[0])
print(mc_fall_data[0])
print(mc_notfall_data[0])

# read target label of UR dataset.

target = []
path1 = '/content/drive/My Drive/636project/target/urfall-cam0-falls.csv'
path2 = '/content/drive/My Drive/636project/target/urfall-cam0-adls.csv'
# in these csv of UR dataset,
# '-1' means person is not lying, '1' means person is lying on the ground; '0' is temporary pose, when person "is falling"

def input_fall(path):
  with open(path, mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
      record = []
      record.append(row[0])
      record.append(row[1])
      label = row[2]
      if label == '1' or label == '0':
        record.append(1)
      else:
        record.append(0)
      # each record: [<video_id>, <frame_id>, label], eg: ['fall-17', '22', 0]
      target.append(record)

# though laying in the video, but it is not fall, so I mark 0 as label
def input_adl(path):
  with open(path, mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
      record = []
      record.append(row[0])
      record.append(row[1])
      record.append(0)
      target.append(record)

input_fall(path1)
print("falls of UR dataset", len(target))

input_adl(path2)

df = pd.DataFrame.from_records(target)
df.to_csv (r'/content/drive/My Drive/636project/target_data.csv', index = False, header=False)

# make dictionary of target of ur dataset
target_csv_path = '/content/drive/My Drive/636project/target_data.csv'
idx = 0
idx_dict = {}

with open(target_csv_path, mode='r') as csv_file:
  reader = csv.reader(csv_file)
  for row in reader:
    # each row: [<video_id>, <frame_id>, label], eg: ['fall-17', '22', 0]
    idx_dict['.'.join(row[:2])] = idx  # {'fall-17.22' : 112}
    idx += 1

# match body landmark data with label data of ur dataset
bodylandmark = []
label = []
frame_name = []

for landmark in ur_data:
  video_name = landmark[-1].split('_')  # 'fall-01-cam0_000000000004'
  a = video_name[0].split('-')
  video_type = a[0] # 'fall'
  video_id = a[1]   # '01'
  frame_id = str(int(video_name[1])) # '000000000004' become '4'
  try:
    a = '.'.join([video_id, frame_id])
    b = '-'.join([video_type, a])
    label.append(target[idx_dict[b]][-1])
    bl = landmark[:-1]
    # convert string into float
    bl = list(map(float,bl))
    bodylandmark.append(bl)
    frameName = landmark[-1]
    frame_name.append(frameName)
  except:
    continue

print("bodylandmark record sample of ur dataset:" )
print(bodylandmark[0])
print("label of ur dataset: ", len(label))
print("bodylandmark of ur dataset: ", len(bodylandmark))
print(frame_name[0])

# add data of MC dataset to the whole dataset
# For MC dataset, all frames in fall data should be label 1,
# and all frames in notfall data should be label 0.
for record in mc_fall_data:
  bl = record[:-1]
  bl = list(map(float,bl))
  bodylandmark.append(bl)
  label.append(1)
  frameName = record[-1]
  frame_name.append(frameName)

for record in mc_notfall_data:
  bl = record[:-1]
  bl = list(map(float,bl))
  bodylandmark.append(bl)
  label.append(0)
  frameName = record[-1]
  frame_name.append(frameName)

# normalize data
for record in bodylandmark:
  for i in range(0, 63, 3):
    record[i] = float(record[i])/ 640
    record[i + 1] = float(record[i + 1]) / 480

print(bodylandmark[0])
print(len(label))
print(len(bodylandmark))

# make bodylandmark and label to be a same data frame
all_data = []

for i in range(len(bodylandmark)):
  all_data.append([label[i]])
  all_data[i] = all_data[i] + bodylandmark[i]

print(all_data[0])
print(len(all_data))
print(len(all_data[0]))

# df = pd.DataFrame.from_records(all_data)
test_cnn = []
test_lstm = []
y_test = []
correct = 0
incorrect = 0
cnn_correct = 0
cnn_incorrect = 0
lstm_correct = 0
lstm_incorrect = 0

for record in all_data:
  test_cnn.append(record[1:])
  test_lstm.append(record[1:])
  label = int(record[0])
  y_test.append(label)

# the bodylandmark directory for a vedio
video_landmark_path = sys.argv[1]

# use model and write the json file for a video
print("loading model and computing probability for each frame...")
cnn_model = keras.models.load_model('/content/drive/My Drive/636project/model_improved_cnn.h5')
lstm_model = keras.models.load_model('/content/drive/My Drive/636project/model_improved_lstm.h5')

# for cnn model, reshape and predict
test_cnn = array(test_cnn)
test_cnn = test_cnn.reshape((len(test_cnn), 3, int(len(test_cnn[0])/3), 1))
probability_cnn = cnn_model.predict_proba(test_cnn)

# for lstm model, reshape and predict
test_lstm = array(test_lstm)
test_lstm = test_lstm.reshape((len(test_lstm), 1, len(test_lstm[0])))
probability_lstm = lstm_model.predict_proba(test_lstm)

# emsemble the prediction of cnn and lstm model
frame_num = 0

for i in range(len(probability_cnn)):
    timestamp = frame_num / 30
    probability_fall = float(probability_cnn[i][0]) * 0.85 + float(probability_lstm[i][0]) * 0.15
  # emsembled model
    if probability_fall < 0.5:
      prediction = 0.0
    else:
      prediction = 1.0

    if prediction == float(y_test[i]):
      correct += 1
    else:
      incorrect += 1
  # cnn model
    if probability_cnn[i][0] < 0.5:
      prediction_cnn = 0.0
    else:
      prediction_cnn = 1.0

    if prediction_cnn == float(y_test[i]):
      cnn_correct += 1
    else:
      cnn_incorrect += 1

  # lstm model
    if probability_lstm[i][0] < 0.5:
      prediction_lstm = 0.0
    else:
      prediction_lstm = 1.0

    if prediction_lstm == float(y_test[i]):
      lstm_correct += 1
    else:
      lstm_incorrect += 1

print('finish prediction')

print('emsembled model')
print(correct)
print(incorrect)
print(correct / (correct + incorrect))

print('emsembled model')
print(correct)
print(incorrect)
print(correct / (correct + incorrect))

print('cnn model')
print(cnn_correct)
print(cnn_incorrect)
print(cnn_correct / (cnn_correct + cnn_incorrect))

print('lstm model')
print(lstm_correct)
print(lstm_incorrect)
print(lstm_correct / (lstm_correct + lstm_incorrect))
