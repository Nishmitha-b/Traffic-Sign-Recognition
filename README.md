
# Traffic Sign Recognition using ResNet34 and Flask

## Project Overview

This project is a Traffic Sign Recognition System developed using Deep Learning and Transfer Learning techniques. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and can classify traffic signs into multiple categories through a Flask-based web application.

The system helps in automatically identifying important road signs, which can be useful in intelligent transportation systems and autonomous driving applications.

## Features

* Traffic sign image classification
* Transfer Learning using ResNet34
* Data augmentation for improved generalization
* Train, Validation, and Test dataset splitting
* Performance evaluation using multiple metrics
* Flask-based web interface for predictions

## Dataset

Dataset Used: German Traffic Sign Recognition Benchmark (GTSRB)

Selected Classes:

* Speed Limit 30
* Speed Limit 50
* Priority Road
* Give Way
* Stop
* No Entry
* Turn Right
* Road Work

## Technologies Used

* Python
* PyTorch
* Flask
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

## Deep Learning Approach

The project uses a pre-trained ResNet34 architecture and fine-tunes the final classification layer for traffic sign recognition.

### Data Processing

* Image resizing
* Image normalization
* Data augmentation
* Dataset balancing

### Model Evaluation

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

## Project Structure

Traffic-Sign-Recognition/

├── app.py

├── Traffic_sign.ipynb

├── traffic_sign_model.pth

├── static/

├── templates/

└── README.md

## Future Improvements

* Real-time webcam traffic sign detection
* Support for all GTSRB traffic sign categories
* Deployment on cloud platforms
* Integration with autonomous driving systems

## Author

Nikitha N Pavar

