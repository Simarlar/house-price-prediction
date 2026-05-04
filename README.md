🏠 House Price Prediction – End-to-End MLOps Project
📌 Overview

This project is a complete Machine Learning Operations (MLOps) pipeline for predicting house prices using regression models. It includes data processing, model training, experiment tracking, API deployment, and a full-stack web application.

The system is designed with a microservices architecture:

🧠 Machine Learning Model (Regression)
⚙️ FastAPI Backend (Model Serving)
🌐 Flask Frontend (UI)
🐳 Dockerized Deployment
📊 Feature Engineering + ML Pipeline
🚀 Live Architecture
User → Flask UI → FastAPI Model API → ML Model → Prediction
🧠 Problem Statement

Predict house prices based on structural and location-based features such as:

Square Footage
Number of Bedrooms
Number of Bathrooms
Year Built
Lot Size
Garage Size
Neighborhood Quality
📊 Dataset Features
Feature	Description
Square_Footage	Total living area
Num_Bedrooms	Number of bedrooms
Num_Bathrooms	Number of bathrooms
Year_Built	Construction year
Lot_Size	Land size
Garage_Size	Garage capacity
Neighborhood_Quality	Rating (1–5)
House_Price	Target variable
⚙️ Feature Engineering

Advanced features created during transformation:

House Age
Square Footage per Bedroom
House Density
Luxury Score (Interaction Feature)
🤖 Machine Learning Models Used
Linear Regression
Random Forest Regressor

Best model selected using:

R² Score
MAE
📈 Model Tracking

Experiment tracking is managed using:

MLflow

Tracked metrics:

Accuracy (R² Score)
Error Metrics ( MAE)
Model parameters
Model artifacts
🧱 Tech Stack
Backend
FastAPI
Pydantic
Scikit-learn
Pandas
Joblib
Frontend
Flask
HTML/CSS
MLOps
MLflow
Docker
Docker Compose
Deployment
Render
🐳 Docker Setup
Build Images
docker build -t house-price-api -f Dockerfile.api .
docker build -t house-price-flask -f Dockerfile.flask .
Run with Docker Compose
docker compose up --build
🌐 API Endpoints
Health Check
GET /
Prediction Endpoint
POST /predict
Example Request
{
  "Square_Footage": 2000,
  "Num_Bedrooms": 3,
  "Num_Bathrooms": 2,
  "Year_Built": 2010,
  "Lot_Size": 0.25,
  "Garage_Size": 2,
  "Neighborhood_Quality": 4
}
Example Response
{
  "prediction": 245000.75
}
🖥️ Frontend UI
Simple Flask-based web UI
Real-time prediction display
Form validation included
Clean responsive design
📁 Project Structure
src/
 ├── components/
 ├── pipeline/
 ├── entity/
 ├── logger.py
 ├── exception.py

app.py                  # Flask UI
api.py                  # FastAPI service
artifacts/              # Model + preprocessor
Dockerfile.api
Dockerfile.flask
docker-compose.yml
🔄 MLOps Workflow
Data Ingestion
     ↓
Data Validation
     ↓
Data Transformation + Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
MLflow Tracking
     ↓
Dockerization
     ↓
Deployment (Render)
📦 How to Run Locally
1. Clone Repo
git clone https://github.com/your-repo.git
cd house-price-prediction
2. Install Dependencies
pip install -r requirements.txt
3. Run FastAPI
uvicorn api:app --reload
4. Run Flask
python app.py
☁️ Deployment Notes

When deploying on Render:

Set environment variable in Flask:
API_URL = https://your-fastapi-service.onrender.com/predict
📌 Key Learnings
End-to-end ML pipeline design
Feature engineering for regression
Model tracking using MLflow
API development with FastAPI
Frontend integration with Flask
Docker containerization
Microservices architecture
🔥 Future Improvements
Kubernetes deployment
Monitoring with Prometheus/Grafana
Cloud storage for models (S3)
👨‍💻 Author

Simarlar

⭐ If you like this project

Give it a star ⭐ on GitHub and feel free to fork!