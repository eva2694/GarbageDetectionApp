# Garbage Detection App

## Overview

The Garbage Detection App is designed to monitor resource usage and model performance on Android devices. This app leverages advanced object detection algorithms such as YOLOv8 and EfficientDet-Lite to provide real-time detection and classification of garbage. It helps in evaluating which model performs best in terms of accuracy, inference time, CPU usage, memory consumption, and battery usage.

## Features

- **Real-Time Detection**: Provides instant detection and classification of garbage from live camera feed.
- **Multiple Models**: Supports multiple pre-trained models, including YOLOv11 and EfficientDet-Lite.
- **Detailed Metrics**: Displays detection time, CPU usage, frame latency, frames per second (FPS), and average confidence.
- **Battery and Memory Monitoring**: Keeps track of available and total memory, as well as battery level during detection.

## Models

### YOLOv11
- **YOLOv11n**: Lightweight model with high accuracy.
- **YOLOv11s**: Slightly larger model with higher accuracy.
- **YOLOv11m**: Larger model with even higher accuracy.

### EfficientDet-Lite
- **EfficientDet-Lite0**: Optimized for faster inference.
- **EfficientDet-Lite2**: Slightly more complex, providing better accuracy at the cost of inference speed.
- **EfficientDet-Lite4**: Even more complex, providing even better accuracy at the cost of inference speed.

## Training and Evaluation

The models were trained and evaluated using the [GARBAGE CLASSIFICATION Dataset](https://universe.roboflow.com/material-identification/garbage-classification-3), which contains 10 000 images of garbage in 6 classes: Biodegradable, Glass, Plastic, Metal, Cardboard and Paper.

### Training Details

- **YOLOv11 Models**: Trained using the Ultralytics framework.

- **EfficientDet-Lite Models**: Trained using TensorFlow Lite Model Maker.

### Evaluation Results


### Example Images

#### App Interface



## How to Use

1. **Launch the App**: Open the app and grant camera permissions.
2. **Select Model**: Choose the desired model from the dropdown list.
3. **Start Detection**: Begin real-time detection by pointing your camera at garbage.
4. **View Results**: Observe the detection results along with performance metrics on the screen.
5. **DATA**: Get the data from the csv file. It will be generated in the emulated storage on your android device. Path : `/storage/emulated/0/Android/data/package.name
/files`.
*(Aditional resources usage data: Profiler and AGI.)*

## How to Install the Garbage Detection App

1. **Download the APK**:
   - [Download the app](tbd) from OneDrive.

2. **Enable Installation from Unknown Sources**:
   - Go to `Settings` > `Security`.
   - Enable `Install unknown apps` or `Unknown sources`.
   - If you are using Android 8.0 or higher, you may need to grant this permission to the app from which you are installing the APK.

3. **Install the APK**:
   - Open the downloaded APK file.
   - Follow the on-screen instructions to complete the installation.

4. **Open the App**:
   - Once installed, open the app.

## Data Analisys

## Acknowledgements

- [Roboflow](https://roboflow.com) 
- [Ultralytics](https://ultralytics.com) 
- [SLING](https://www.sling.si/en/)
- [TFLite](https://www.tensorflow.org/lite/android)
