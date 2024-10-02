<img width="685" src="https://github.com/user-attachments/assets/1136c028-4254-4e01-830b-f57b5d2a4794" alt="Screenshot 2024-09-26 at 8 19 45 PM">  </br>
# 🚀 **Suk AI**: Real-Time Object Detection & Image Classification System

In this project, we will explore how **Suk AI**, an advanced real-time object detection and image classification system, was built using **YOLOv5** and **NVIDIA AI Workbench**. The system can be deployed across various industries such as retail, security, and agriculture, providing automated solutions for object detection and classification in real-time environments.

---

## 🛠️ **Project Overview**

**Suk AI** was developed with the goal of providing efficient real-time object detection and classification across multiple domains. Leveraging GPU acceleration through **NVIDIA AI Workbench**, the model offers fast and accurate results, making it ideal for use cases like:

- **Retail**: Detect products on shelves to streamline inventory management.
- **Security**: Monitor video feeds for unusual objects or behaviors.
- **Agriculture**: Classify and assess crops for health monitoring.

---

## 📚 **Table of Contents**
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Real-Time Inference](#real-time-inference)
- [Challenges We Encountered](#challenges-we-encountered)
- [Future Plans](#future-plans)
- [Conclusion](#conclusion)

---

## 📥 **Data Preparation**

Data is the foundation of any machine learning model. For **Suk AI**, we used the [COCO Dataset](https://cocodataset.org/) to train the object detection and image classification models.

### Key Steps:
- **Dataset**: We used the COCO dataset to cover diverse object categories.
- **Data Cleaning**: Cleaned the dataset to remove noise and ensure consistency.
- **Labeling**: Ensured proper bounding box annotations following the YOLO format.

> **Note**: You can also use your own datasets depending on your domain. Proper annotation and labeling are key to effective model training.

---

## 🔧 **Model Training**

We leveraged **YOLOv5** for object detection and **ResNet50** for image classification, all of which were trained using **NVIDIA AI Workbench** for GPU acceleration.

1. **Model Selection**: Started with **YOLOv5s** for its balance between speed and accuracy.
2. **Transfer Learning**: Fine-tuned the pre-trained YOLOv5 weights on our custom dataset.
3. **Training Parameters**:
   - **Epochs**: 25 for demonstration purposes, but more (100+) are recommended for real-world tasks.
   - **Batch Size**: 16 for training to balance speed and memory efficiency.

> **Tip**: Use pre-trained weights to save training time and improve accuracy. Transfer learning helps the model adapt quickly to new datasets.

---

## 🎥 **Real-Time Inference**

Once the model was trained, we deployed it for real-time inference using **OpenCV**. The model processes live video streams to detect and classify objects in real time.

- **Integration**: We integrated with **OpenCV** to process live video feeds, making it suitable for applications such as surveillance or automated product recognition.
- **Results**: Bounding boxes and class labels are displayed on the video stream for visual feedback.

---

## 🏗️ **Challenges We Encountered**

Building **Suk AI** came with a number of challenges:
- **Data Quality**: Cleaning and preparing large datasets was time-consuming but crucial to ensure model performance.
- **Balancing Speed and Accuracy**: Fine-tuning the model to maintain low latency while ensuring high accuracy in real-time environments was a key challenge.
- **Real-Time Integration**: Ensuring that the object detection model could run efficiently on video streams without performance bottlenecks was vital for success.

---

## 🎉 **Achievements**

We are proud of several key accomplishments in the development of **Suk AI**:
- Built a real-time object detection and image classification system capable of handling large datasets.
- Achieved high accuracy while maintaining low latency in real-time video feeds.
- Successfully integrated the system into live video processing using **OpenCV**.

---

## 📚 **What We Learned**

This project provided valuable insights into:
- **Transfer Learning**: We learned how to fine-tune pre-trained models to adapt to our custom dataset.
- **Real-Time Processing**: Optimized the system for performance in real-time applications using GPU acceleration.
- **Data Management**: Learned the importance of cleaning and balancing large datasets to ensure effective model training.

---

## 🚀 **Future Plans**

We have exciting plans to expand **Suk AI** further:
1. **Multi-Object Tracking**: Adding the ability to track multiple objects over time in video streams for improved security and surveillance applications.
2. **Edge Deployment**: Optimizing the system for deployment on edge devices, such as drones and security cameras, for greater scalability.
3. **Custom Applications**: Expanding into industries like healthcare (for anomaly detection), logistics (for product identification in warehouses), and autonomous vehicles (for real-time object detection and navigation).

---

## 🎯 **Conclusion**

**Suk AI** represents a cutting-edge solution for real-time object detection and image classification across multiple industries. By harnessing the power of **YOLOv5** and **NVIDIA GPUs**, we've built a system that is fast, accurate, and scalable. With further advancements and optimizations, **Suk AI** can expand into numerous specialized use cases, revolutionizing automation across domains.

Thank you for following along in this journey to build **Suk AI**! Stay tuned for future updates as we continue to refine and scale the project.

<img src="https://github.com/user-attachments/assets/91759712-495e-4941-b466-c6c59ac6fa01" alt="Screenshot 2024-10-02 at 8 26 15 PM">
<img src="https://github.com/user-attachments/assets/ab716270-8e71-4313-8484-4fccd85feffb" alt="Screenshot 2024-10-02 at 7 56 31 PM">
<img src="https://github.com/user-attachments/assets/fbc8c890-4c60-453b-8f20-4e2e9e62244f" alt="Screenshot 2024-10-02 at 7 56 22 PM">

<img src="https://github.com/user-attachments/assets/d2a69346-6190-4240-91b2-f4b0a2be134c" alt="Screenshot 2024-10-02 at 7 56 16 PM">
<img src="https://github.com/user-attachments/assets/17c00334-1553-4164-8397-a073299d1850" alt="Screenshot 2024-10-02 at 7 05 14 PM">
<img src="https://github.com/user-attachments/assets/6905fedd-8ae0-4192-9a83-c562339e8c42" alt="Screenshot 2024-10-02 at 7 29 34 PM">
<img src="https://github.com/user-attachments/assets/ee84c698-c56a-4b29-a91d-c71613bda290" alt="Screenshot 2024-10-02 at 7 31 58 PM">
<img src="https://github.com/user-attachments/assets/0ec354ed-f23d-4fc0-aeab-a9efc8f79a48" alt="Screenshot 2024-10-02 at 7 55 52 PM">
<img src="https://github.com/user-attachments/assets/a1a180cf-1bce-4883-b729-cf52d6c06b16" alt="Screenshot 2024-10-02 at 7 55 42 PM">
<img src="https://github.com/user-attachments/assets/7264d29e-4226-4a42-9b74-64c297a804db" alt="Screenshot 2024-10-02 at 7 55 59 PM">





