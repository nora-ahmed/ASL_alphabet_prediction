#English Sign Language Prediction with MobileNetV2 

This project demonstrates the use of a pre-trained **MobileNetV2** model to classify English Sign Language using images. The model was trained using **TensorFlow** and **Keras**, finally using **OpenCV** to detect hand signs and predict the corresponding letter or symbol.
---
##Project Overview  

The model is trained on a dataset containing images of English Sign Language gestures. Using a pre-trained **MobileNetV2** model, we fine-tuned the model to classify 29 distinct signs corresponding to the letters A-Z and three additional classes: SPACE, DELETE, and NOTHING. The project includes data preprocessing and model training
---
###Technologies Used
- **TensorFlow & Keras for model building**
- **MobileNetV2 as the pre-trained model**
- **pandas and numpy for data handling**
- **OpenCV for real-time detection**
---
##Installation  

To run this project on your local machine, follow these steps:
---
###Prerequisites  

Make sure you have the following installed:
- visual studio code
- Python 3.7+
- pip (Python package manager)
- Git
---
###Clone the repository  

\```python  

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/nora-ahmed/ASL_alphabet_prediction.git)  
\```    

\```python  

cd ASL_alphabet_prediction
\```
---
###Install dependencies  

Required Libraries  

\```python
pip install tensorflow pandas numpy
\```
---
###Running the project  

To run the real-time hand sign detection.  

\```python
python main.py
\```  

This will open a window where you can detect hand signs using your camera. (make sure you are using a laptop or use any camera it will be fine)
---
##How it works
- **Model:** We used MobileNetV2 as a base model and added custom dense layers on top for classification.
- **Training:** The model was trained using images of hand signs, split into training, validation, and test sets.
---
##Team Members
1. Nora Ahmed
2. Youssef Ahmed
3. Yousab Tanious 
