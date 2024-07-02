# Pneumonia-Detection-using-CNN
## Neural Networks and Deep Learning

I. Introduction: 
    Pneumonia is a critical global health concern, demanding early and accurate detection for effective treatment. This project introduces a novel method for pneumonia detection 
    using Convolutional Neural Networks (CNNs). Leveraging CNNs, the model has been trained on a substantial dataset of chest X-ray images and evaluated on a separate test set. 
    The goal is to improve accuracy and efficiency in diagnosing pneumonia, a potentially life-threatening lung infection affecting millions annually.

II. Model Objective:
    The primary objective is to develop an automated pneumonia detection system using CNNs. The model is designed to analyze chest X-ray images, discerning the presence of pneumonia 
    with high accuracy. Early diagnosis is paramount for timely intervention and prevention of complications. The CNN-based approach provides a promising solution to enhance pneumonia detection,
    aiding healthcare professionals in their decision-making process.

III. Dataset Description:
    The dataset consisted of a total of 5863 Chest X-ray images belonging to 2 classes - Pneumonia / Normal.
    
    1. Train Dataset - 5216 Images (Pneumonia :3875 + Normal: 1341)
    2. Test Dataset - 624 Images   (Pneumonia: 390 + Normal: 234 )
    3. Validation Dataset - 16 Images (Pneumonia: 8 + Normal: 8)

IV. Challenges Addressed:
    The project addresses several key challenges in pneumonia detection:

    1. Data Pre-processing: Standardizing image sizes, pixel normalization, and data augmentation to prepare the dataset for training.
    2. Handliong class imbalace: Oversampled normal images dataset to equalize with Pneumonia dataset.
    3. Transfer learning: Designing an effective CNN architecture that can learn and extract relevant features from X-ray images.
    4. Evaluation Metrics: Establishing robust evaluation metrics to accurately assess the model's performance.
    5. k-fold cross validation and Hypertuning: Used to assess the performance and generalization ability of a machine learning model.  
       These are configuration variables that govern the training process and directly impact the performance of the model.
    6. Bootstrap Aggregating:
