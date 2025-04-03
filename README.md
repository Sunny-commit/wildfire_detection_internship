
# Wildfire Detection  
This project focuses on wildfire detection using deep learning using tensorflow. It utilizes the **Wildfire Dataset** from Kaggle to build and test models for detecting wildfires from images.  

## Dataset  
The dataset is downloaded using `kagglehub` and moved to the appropriate directory for processing.  

## Installation  
To run this project, install the required dependencies using:  

pip install kagglehub shutil


## Usage  
1. Download the dataset using:  

   import kagglehub
   path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")

2. Copy the dataset to the content folder:  

   import shutil
   import os
   shutil.copytree('/kaggle/input/the-wildfire-dataset', '/content/the-wildfire-dataset', dirs_exist_ok=True)

3. Run the notebook to train the model and analyze results.  

## Dependencies  
- Python  
- KaggleHub  
- shutil  
- os  

## Credits  
This project is developed as part of the **Edunet Foundation Internship**.  


