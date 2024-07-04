# Pneumonia Detection using Deep Learning
## Neural Networks and Deep Learning

I. Introduction: 
   Pneumonia poses a significant global health challenge, necessitating precise and prompt identification for effective medical intervention. This project introduces an innovative approach 
   to detect pneumonia using Convolutional Neural Networks (CNNs). By harnessing the power of CNNs, the model has been trained on a large dataset comprising chest X-ray images and rigorously 
   assessed using an independent test dataset. The objective is to enhance the precision and effectiveness of pneumonia diagnosis, addressing a critical need in the diagnosis of this severe 
   lung infection that impacts millions of people each year.
   
II. Model Objective:
    The main goal is to create a CNN-based system for automated pneumonia detection. This model is specifically engineered to analyze chest X-ray images and accurately identify the presence 
    of pneumonia. Early detection is crucial for timely treatment and preventing complications. By leveraging CNN technology, this approach offers a promising solution to improve pneumonia 
    detection, supporting healthcare professionals in making informed decisions.

III. Dataset Description:
    The dataset consisted of a total of 5863 Chest X-ray images belonging to 2 classes - Pneumonia / Normal.
    
   1. Train Dataset - 5216 Images (Pneumonia :3875 + Normal: 1341)
   2. Test Dataset - 624 Images   (Pneumonia: 390 + Normal: 234 )
   3. Validation Dataset - 16 Images (Pneumonia: 8 + Normal: 8)

IV. Requirements:
    The following libraries are required to run this project.
    1. Python 3.18
    2. PyTorch
    3. torchvision
    4. matplotlib
    5. numpy
    6. PIL
    
V. Challenges Addressed:
    The project addresses several key challenges in pneumonia detection:

   1. Data Pre-processing: Standardizing image sizes, pixel normalization, and data augmentation to prepare the dataset for training.
   2. Handliong class imbalace: Oversampled normal images dataset to equalize with Pneumonia dataset.
   3. Transfer learning: Designing an effective CNN architecture that can learn and extract relevant features from X-ray images.
   4. Evaluation Metrics: Establishing robust evaluation metrics to accurately assess the model's performance.
   5. k-fold cross validation and Hypertuning: Used to assess the performance and generalization ability of a machine learning model.  
       These are configuration variables that govern the training process and directly impact the performance of the model.
   6. Bootstrap Aggregating: The idea is to train models on slightly different datasets to capture different aspects of the data.
