# -*- coding: utf-8 -*-
"""Copy of 636model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16y1ZtmkBpeJCvto1H8OPCkje9GUg-_wf
"""

import keras
keras.__version__

from keras import models
from keras import layers

import numpy as np
import json
import os
import csv
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# read json 2-d pose as data, without using: 
#     {15, "REye"},
#     {16, "LEye"},
#     {17, "REar"},
#     {18, "LEar"}

basepath1 = '/content/drive/My Drive/636project/data/fall_UR/fall-cam0'
basepath2 = '/content/drive/My Drive/636project/data/adl_UR'

def input_data(basepath):
  data = []
  entries = os.listdir(basepath)
  for entry in entries:
    path = os.path.join(basepath, entry)
    frames = os.listdir(path)
    for frame in frames:
      frame_path = os.path.join(path, frame)
      with open(frame_path, mode='r') as json_file:
        people_dict = json.load(json_file)
        people = people_dict["people"]
        pose_keypoints_2d = []
        # fill missing data as 0
        if len(people) == 0:
          pose_keypoints_2d = [0] * 63
        else:
          full_pose = people[0].get("pose_keypoints_2d")
          pose_keypoints_2d = full_pose[:45] + full_pose[57:]
         
        pose_keypoints_2d.append(frame.split('.')[0])
        # each pose_keypoints_2d: 
        # [431.949, 196.241, 0.0564434, 437.194, 187.749, 0.552267,...'fall-06-cam0_000000000065_keypoints']
        data.append(pose_keypoints_2d)        
  return data

  
data1 = input_data(basepath1)
print(len(data1))

data2 = input_data(basepath2)
print(len(data2))

video_data = data1 + data2
print(len(video_data))

df = pd.DataFrame.from_records(video_data)
df.to_csv (r'/content/drive/My Drive/636project/raw_data.csv', index = False, header=False)

video_data = []
path_of_video = '/content/drive/My Drive/636project/raw_data.csv'

# read the video data csv,
# it is faster than run the above code each time
def read_data(path):
  with open(path, mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
      video_data.append(row)

read_data(path_of_video)

# target: csv file
target = []
path1 = '/content/drive/My Drive/636project/target/urfall-cam0-falls.csv'
path2 = '/content/drive/My Drive/636project/target/urfall-cam0-adls.csv'
# in these csv of UR dataset, 
# '-1' means person is not lying, '1' means person is lying on the ground; '0' is temporary pose, when person "is falling"

def input_fall(path):
  falls = 0
  not_fall = 0
  with open(path, mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
      record = []
      record.append(row[0])
      record.append(row[1])
      label = row[2]
      if label == '1' or label == '0':
        falls += 1
        record.append(1)        
      else:
        not_fall += 1
        record.append(0)
      # each record: [<video_id>, <frame_id>, label], eg: ['fall-17', '22', 0] 
      target.append(record)
  print("falls", falls)
  print("not_fall", not_fall)

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
print(len(target))

input_adl(path2)
print(len(target))

df = pd.DataFrame.from_records(target)
df.to_csv (r'/content/drive/My Drive/636project/target_data.csv', index = False, header=False)

# make dictionary of target 
target_csv_path = '/content/drive/My Drive/636project/target_data.csv'
idx = 0
idx_dict = {}

with open(target_csv_path, mode='r') as csv_file:
  reader = csv.reader(csv_file)
  for row in reader:
    # each row: [<video_id>, <frame_id>, label], eg: ['fall-17', '22', 0] 
    idx_dict['.'.join(row[:2])] = idx  # {'fall-17.22' : 112}
    idx += 1

bodylandmark = []
label = []

for landmark in video_data:
  video_name = landmark[-1].split('_')  # 'fall-01-cam0_000000000004'
  a = video_name[0].split('-')
  video_type = a[0] # 'fall'
  video_id = a[1]   # '01'
  frame_id = str(int(video_name[1])) # '000000000004' become '4'
  try:
    a = '.'.join([video_id, frame_id])
    b = '-'.join([video_type, a])
    label.append(target[idx_dict[b]][-1])
    bodylandmark.append(landmark[:-1])
  except:
    continue

print(bodylandmark[0])
print(len(label))
print(len(bodylandmark))

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

# up-sampling to make the data balance
# Separate majority and minority classes
df = pd.DataFrame.from_records(all_data)
header = ['label']
for i in range(63):
  header.append(i)

df.columns = header
df_majority = df[df.label==0]
df_minority = df[df.label==1]
print("before re-sampling, fall vs not fall: ")
print(df['label'].value_counts())

from sklearn.utils import resample
print("begin to re sample...")
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,      # sample with replacement
                                 n_samples=9741)    # to match majority class
                                  
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
print("after re-sampling...")
# Display new class counts
df_upsampled.label.value_counts()

all_data = df_upsampled.values

print(all_data[0])
print(len(all_data))
print(len(all_data[0]))

# split data into train and test set
np.random.shuffle(all_data)
split_point = len(all_data) // 9
print(split_point)
test_data = all_data[:split_point]
train_data = all_data[split_point: ]

print(len(test_data))
print(len(train_data))

#train model using lstm

from keras.layers import LSTM
from keras import callbacks
from keras import layers
from numpy import array
from keras.models import Sequential

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * np.exp(0.1 * (10 - epoch))

x_train = []
y_train = []

for record in train_data:
  x_train.append(record[1:])
  label = int(record[0])
  y_train.append(label) 

x_train = array(x_train)
x_train = x_train.reshape((len(x_train), 1, len(x_train[0])))

model = Sequential()
model.add(LSTM(32, input_shape=(1, 63)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5),
                    callbacks.LearningRateScheduler(scheduler)])

model.summary()

# plotting the results

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# train the final model on all non-test data avaliable

x_test = []
y_test = []

for record in test_data:
  x_test.append(record[1:])
  label = int(record[0])
  y_test.append(label) 

x_test = array(x_test)
x_test = x_test.reshape((len(x_test), 1, len(x_test[0])))

test_score = model.evaluate(x_test, y_test)
print("test loss:")
print(test_score[0])
print("test accuracy:")
print(test_score[1])

model.save("/content/drive/My Drive/636project/model.h5")

