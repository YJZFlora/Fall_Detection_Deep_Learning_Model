# Fall_Detection_Deep_Learning_Model
falling detection model using Deep Learning models: LSTM and 2D CNN.

## model file
model_improve_lstm.h5

model_improve_cnn.h5

## Performace of models

LSTM model:

Loss 0.1497     Accuracy 0.9395

2D CNN model:

Loss 0.0667     Accuracy 0.9692

## How to use the model?
video demo: https://youtu.be/r2CNC9QNPMg

1. Clone this repository: https://github.com/YJZFlora/Fall_Detection_Deep_Learning_Model

2. Turn to the directory:
```cd …/YJZFlora/Fall_Detection_Deep_Learning_Model```

3. Install necessary libraries:
* python 3
* Keras
* Tensorflow
* pandas
* Numpy

4. Generate body landmark json files, copy the directory of it.

I used the open source tool by CMU-Perceptual-Computing-Lab(https://github.com/CMU-Perceptual-Computing-Lab/openpose) to generate body landmark.

Sample body landmark files have been included in samples directory.

eg, Fall_Detection_Deep_Learning_Model/samples/bodylandmark/test_case1

5. Run one of the following scripts:

The file name or directory name should not include under_score "_"

Execute emsembled model:
``` python3 execute_model_ensembled.py <width of the video> <height of the video> <directory of body landmarks for a video>```

Execute new LSTM model:
```python3 execute_model_lstm.py <width of the video> <height of the video> <directory of body landmarks for a video>```

Execute CNN model:
```python3 execute_model_cnn.py <width of the video> <height of the video> <directory of body landmarks for a video>```

eg:
```python3 execute_model_ensembled.py 640 360 ./test/1800-3```

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
