# Autonomy in the Fields – Life-Saving Computer Vision

## Traktoros

by Team undark.jena

### 1. Introduction

This project was developed at the HackHPI 2026. The goal of the challenge is to develop robust computer vision systems that can detect critical objects and situations in agricultural environments to improve safety and enable autonomous decision-making.

Our project focuses on building a web-based platform that demonstrates an AI-powered detection pipeline for agricultural self-driving vehicles, where rapid identification of hazards is necessary to prevent accidents.

### 2. System Overview

Our website acts as a demonstration interface for our computer vision pipeline, allowing users to analyze field images and visualize detection results.

The platform consists of three main components:

#### 2.1 Frontend Interface

The frontend provides a simple and accessible interface where users can:

- Upload images from agricultural environments
- Trigger model inference
- View detection results with bounding boxes and labels

Key design goals:

- Simplicity
- Fast interaction
- Clear visualization of AI predictions

#### 2.2 Backend Processing

The backend handles the core processing pipeline:

1. Image upload and preprocessing
2. Running the computer vision model
3. Post-processing predictions
4. Returning structured detection results

The server processes images and returns the detected objects along with:

- Confidence scores
- Bounding boxes
- Predicted class labels

#### 2.3 Detection Model

The core of the system is a deep learning model trained for object detection.

The model identifies objects relevant to agricultural safety such as:

- Humans
- Equipment
- Potential hazards
- Environmental objects

Modern deep learning models allow systems to automatically learn features such as shapes, edges, and textures from large datasets, improving detection accuracy in complex real-world environments.

### 3. Evaluation

To assess the performance of our computer vision system, we evaluated how accurately the model detects and localizes objects in agricultural field images. Because the system is designed for safety-critical environments, reliable object detection is essential.

We used standard object detection metrics including precision, recall, and Intersection over Union. Precision measures the proportion of correct detections among all predicted detections, while recall measures the proportion of actual objects that the model successfully detects. IoU evaluates how closely the predicted bounding boxes match the ground truth annotations.

Model predictions were compared against labeled validation data that was not used during training. This allowed us to measure how well the model generalizes to unseen images. Detections were considered correct when the overlap between predicted and ground-truth bounding boxes exceeded a predefined IoU threshold.

In addition to quantitative metrics, we performed qualitative evaluation by visually inspecting predictions through the website interface. This helped verify that the model correctly identifies objects under different field conditions such as varying lighting, background complexity, and object sizes.

Overall, the evaluation demonstrates the system’s ability to detect relevant objects and highlights its potential use in agricultural safety monitoring and autonomous field operations.

### 4. Future Vision

While the current prototype demonstrates the concept, several improvements could enhance the system:

- Training on larger agricultural datasets  
- Real-time video processing
- Multi-sensor integration (vision + LiDAR)

These enhancements could allow the system to operate reliably in real-world agricultural environments.

### 5. Conclusion

This project demonstrates how computer vision and web technologies can be combined to create accessible AI systems for safety-critical environments.

Our website provides an interactive platform to showcase how AI models can detect important objects and hazards in agricultural scenes. With further development and training on larger datasets, such systems could support autonomous machines and significantly improve safety in the field.
