# Fall_Detection_Deep_Learning_Model
falling detection model using LSTM

## model file
model.h5

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
4. get the directory of body landmark json files, copy it. (Some generated body landmark files has been included in /samples
 
eg, Fall_Detection_Deep_Learning_Model/samples/bodylandmark/adl-01-cam0

5. run:
Execute new LSTM model:
```python3 execute_model_lstm.py <directory of body landmarks for a video>```

Execute CNN model:
```python3 execute_model_cnn.py <directory of body landmarks for a video>```


eg:
```python3 execute_model.py Fall_Detection_Deep_Learning_Model/samples/bodylandmark/adl-01-cam0```

