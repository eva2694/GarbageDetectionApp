# Road Sign Detection App

## Overview

The Road Sign Detection App is designed to monitor resource usage and model performance on Android devices. This app leverages advanced object detection algorithms such as YOLOv8 and EfficientDet-Lite to provide real-time detection and classification of road signs. It helps in evaluating which model performs best in terms of accuracy, inference time, CPU usage, memory consumption, and battery usage.

## Features

- **Real-Time Detection**: Provides instant detection and classification of road signs from live camera feed.
- **Multiple Models**: Supports multiple pre-trained models, including YOLOv8 and EfficientDet-Lite.
- **Detailed Metrics**: Displays detection time, CPU usage, frame latency, frames per second (FPS), and average confidence.
- **Battery and Memory Monitoring**: Keeps track of available and total memory, as well as battery level during detection.

## Models

### YOLOv8
- **YOLOv8n**: Lightweight model with high accuracy.
- **YOLOv8s**: Slightly larger model with even higher accuracy.

### EfficientDet-Lite
- **EfficientDet-Lite0**: Optimized for faster inference.
- **EfficientDet-Lite1**: Slightly more complex, providing better accuracy at the cost of inference speed.

## Training and Evaluation

The models were trained and evaluated using the [Road Signs Dataset](https://makeml.app/datasets/road-signs), which contains 877 images of road signs in four classes: Traffic Light, Stop, Speed Limit, and Crosswalk. The dataset was preprocessed and augmented using [Roboflow](https://roboflow.com).

### Training Details

- **YOLOv8 Models**: Trained using the Ultralytics YOLOv8 framework.
  - [YOLOv8 Training Notebook](https://colab.research.google.com/dummy_yolo_training_link)

- **EfficientDet-Lite Models**: Trained using TensorFlow Lite Model Maker.
  - [EfficientDet-Lite Training Notebook](https://colab.research.google.com/dummy_efficientdet_training_link)

### Evaluation Results

| Model               | mAP  | Average Inference Time  |
|---------------------|------|-------------------------|
| EfficientDet-Lite0  | 0.638| 1.918 s                 |
| EfficientDet-Lite1  | 0.623| 6.269 s                 |
| YOLOv8n             | 0.911| 9.2 ms                  |
| YOLOv8s             | 0.918| 22 ms                   |

### Example Images

#### App Interface

Example of the app detecting road signs in real-time:

![App Interface](path/to/your/app/interface/image.jpg)

#### Model Predictions from Colab

Results from the YOLOv8 model:

![YOLOv8 Prediction](path/to/your/yolov8/prediction/image.jpg)

Results from the EfficientDet-Lite model:

![EfficientDet-Lite Prediction](path/to/your/efficientdet/prediction/image.jpg)

## How to Use

1. **Launch the App**: Open the app and grant camera permissions.
2. **Select Model**: Choose the desired model from the dropdown list.
3. **Start Detection**: Begin real-time detection by pointing your camera at road signs.
4. **View Results**: Observe the detection results along with performance metrics on the screen.

## Acknowledgements

- [Roboflow](https://roboflow.com) for dataset preprocessing.
- [Ultralytics](https://ultralytics.com) for the YOLOv8 framework.
- [Google Colab](https://colab.research.google.com) for providing the environment to train and evaluate the models.
