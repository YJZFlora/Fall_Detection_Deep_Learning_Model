# Fall_Detection_Deep_Learning_Model
falling detection model using Deep Learning models: LSTM and 2D CNN.

## model file
model_improve_lstm.h5

model_improve_cnn.h5

## Performace of models

LSTM model:

loss 0.1693     accuracy 0.9312

2D CNN model:

loss 0.0713     accuracy 0.9676

## How to use the model?
video demo: https://youtu.be/r2CNC9QNPMg

Instruction:
1. clone this repository
2. turn to the directory
3. install necessary libraries:
* python 3
* Keras
* Tensorflow
* pandas
* Numpy
4. generate body landmark json files, copy the directory of it.

I used the open source tool by CMU-Perceptual-Computing-Lab(https://github.com/CMU-Perceptual-Computing-Lab/openpose) to generate body landmark.

Some generated body landmark files has been included in samples directory.

eg, Fall_Detection_Deep_Learning_Model/samples/bodylandmark/adl-01-cam0

5. run:

Execute new LSTM model:
```python3 execute_model_lstm.py <directory of body landmarks for a video>```

Execute CNN model:
```python3 execute_model_cnn.py <directory of body landmarks for a video>```

Execute emsembled model:
``` python3 execute_model_ensembled.py <directory of body landmarks for a video>```

eg:
```python3 execute_model_ensembled.py ./samples/bodylandmark/adl-01-cam0```

6. Result files

Ensembled model:
./results/timeLabel.json
./results/timeLabel.png

LSTM model:
./results/timeLabel_lstm.json
./results/timeLabel_lstm.png


CNN model:
./results/timeLabel_cnn.json
./results/timeLabel_cnn.png



And in each pair in the file, such as [2.721365,0.23445825932504438], the first number is the time, and the second number is the value of the label your binary-classification model predicts. So the above example shows that at 2.721365 second in the video, the label predicted by your binary-classification model changes to 0.23445825932504438.)
