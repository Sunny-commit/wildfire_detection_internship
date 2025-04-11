🔥 Wildfire Detection using CNN
This project focuses on classifying wildfire images using a Convolutional Neural Network (CNN). The dataset used contains images of fire and no-fire scenarios and is processed and trained using TensorFlow and Keras. The main goal is to assist in early wildfire detection from images using deep learning techniques.

📂 Dataset
The dataset is sourced from Kaggle and includes training, validation, and test sets:

Two classes: fire and no_fire

Automatically downloaded using kagglehub

Organized in the following structure:

bash
Copy
Edit
/the-wildfire-dataset/
  ├── train/
  │   ├── fire/
  │   └── no_fire/
  ├── val/
  └── test/
🧠 Model Architecture
A CNN model built using Keras' Sequential API includes:

3 Convolutional layers with increasing filters (32, 64, 128)

MaxPooling after each Conv layer

Flatten + Dense layers

Dropout to reduce overfitting

Final sigmoid activation for binary classification

python
Copy
Edit
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
⚙️ Preprocessing
Images are resized to 150x150 and normalized (rescaled) using ImageDataGenerator. Batch size is set to 32.

📈 Training
Model is trained using:

Binary Crossentropy Loss

Optimizer: adam

Accuracy as the performance metric

🖼️ Visualization
Images from both fire and no-fire categories are visualized for better understanding and confirmation of dataset correctness.

🚀 How to Run
Ensure Kaggle API credentials are configured (kaggle.json)

Run the Python script or Jupyter Notebook

Dataset will automatically download and prepare

Training and evaluation will be performed on the dataset

📌 Dependencies
Python 3.x

TensorFlow

Matplotlib

Kagglehub

Numpy

📊 Output
After training, the model can be used to classify whether an image contains wildfire or not.
