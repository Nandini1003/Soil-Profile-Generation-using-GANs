#PhytoProfile  Soil-Profile-Generation-using-GANs


<img src="soil.jpeg" alt="Description" width="500"/>



## Overview
**PhytoProfile** is a web-based application that leverages **Generative Adversarial Networks (GANs)** and **Machine Learning** to generate synthetic soil profiles, classify soil health, and recommend suitable crops. This tool allows users to input key soil parameters (like pH, electrical conductivity, and nutrient content), generate new soil profiles, and receive actionable insights for optimal plant growth.

This project demonstrates the application of deep learning, data scaling, and label encoding in agriculture, making it an ideal showcase for AI/ML and web deployment skills.

---

## Features

1. **Soil Classification**  
   - Classifies soil based on parameters such as pH, EC, Phosphorus, Potassium, Urea, TSP, MOP, Moisture, and Temperature.
   - Recommends the most suitable plant type based on trained ML model.

2. **Soil Profile Generation (GAN)**  
   - Generates synthetic soil profiles to simulate diverse real-world scenarios.
   - Automatically fills input forms with generated values for evaluation.

3. **Soil Health Evaluation**  
   - Evaluates each soil parameter as Good or Bad.
   - Provides a detailed conclusion and actionable insights.

4. **Interactive Web Interface**  
   - User-friendly Flask web interface.
   - Auto-fills generated soil profiles.
   - Clean, modern UI with black-themed styling.

---

## Tech Stack

- **Backend:** Flask (Python)  
- **Machine Learning:** TensorFlow (Keras), Scikit-learn  
- **Data Processing:** NumPy, Pandas  
- **Model Files:** `.h5` Keras models, `.pkl` Scaler and LabelEncoder  
- **Frontend:** HTML, CSS, JavaScript (with auto-fill functionality)  

---

## Project Structure
Soil-Profile-Generation-Using-GANs-main/
│
├── main.py # Flask application
├── classification_model (1).h5
├── generator_epoch_5000 (1).h5
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
├── templates/
│ └── index.html # Frontend template
└── static/
└── soil.jpeg # Project image

## Setup Instructions

1. **Clone the repository**

[git clone <your-repo-url>
cd Soil-Profile-Generation-Using-GANs-main]

2. **Create a virtual environment**

python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          #Windows

3. **Install dependencies**

pip install -r requirements.txt

4. **Run the Flask app**

python main.py

**Usage**

Classify Soil: Enter soil parameters manually and click Classify and Evaluate.

Generate Soil Profile: Click Generate Profile to auto-fill the form with new synthetic soil data.

View Results: The recommended plant type, evaluation for each parameter, and conclusion are displayed below the forms.


## Project Structure

