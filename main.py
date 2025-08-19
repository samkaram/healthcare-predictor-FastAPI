from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import requests
from typing import Optional

# Create FastAPI app
app = FastAPI(title="Healthcare Predictor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.samkaram.net", "https://samkaram.net", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Define the features (same as your Streamlit app)
FEATURES = [
    'Age', 'Billing Amount', 'Room Number', 'Gender_Male', 'Blood Type_A-', 'Blood Type_AB+', 'Blood Type_AB-',
    'Blood Type_B+', 'Blood Type_B-', 'Blood Type_O+', 'Blood Type_O-', 'Medical Condition_Asthma',
    'Medical Condition_Cancer', 'Medical Condition_Diabetes', 'Medical Condition_Hypertension',
    'Medical Condition_Obesity', 'Admission Type_Emergency', 'Admission Type_Urgent',
    'Medication_Ibuprofen', 'Medication_Lipitor', 'Medication_Paracetamol', 'Medication_Penicillin'
]

# Model URL - you'll need to replace this with your actual download URL
MODEL_URL = "https://github.com/samkaram/healthcare-analytics-app/releases/download/v1.0/healthcare_model.joblib"
# Or use Google Drive: "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download"

# Global model variable
model = None

def download_model():
    """Download model file if it doesn't exist locally"""
    model_path = "healthcare_model.joblib"
    
    if not os.path.exists(model_path):
        print("📥 Downloading model file...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise e
    else:
        print("✅ Model file already exists")
    
    return model_path

def load_model():
    global model
    if model is None:
        try:
            # Download model if needed
            model_path = download_model()
            
            # Load the model
            model = joblib.load(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    return model

# Request model
class PredictionRequest(BaseModel):
    age: int = 30
    billing_amount: float = 1000.0
    room_number: int = 101
    gender: str = "Female"
    blood_type: str = "A+"
    medical_condition: str = "Arthritis"
    admission_type: str = "Elective"
    medication: str = "Aspirin"

# Response model
class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[str] = None
    error: Optional[str] = None
    inputs: Optional[dict] = None

def prepare_prediction_data(request: PredictionRequest):
    """Convert request data to model input format"""
    input_data = {}
    input_data['Age'] = request.age
    input_data['Billing Amount'] = request.billing_amount
    input_data['Room Number'] = request.room_number
    
    # Handle gender
    input_data['Gender_Male'] = 1 if request.gender == "Male" else 0
    
    # Handle blood type
    for bt in ['A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']:
        input_data[f'Blood Type_{bt}'] = 1 if request.blood_type == bt else 0
    
    # Handle medical condition
    for mc in ['Asthma', 'Cancer', 'Diabetes', 'Hypertension', 'Obesity']:
        input_data[f'Medical Condition_{mc}'] = 1 if request.medical_condition == mc else 0
    
    # Handle admission type
    input_data['Admission Type_Emergency'] = 1 if request.admission_type == "Emergency" else 0
    input_data['Admission Type_Urgent'] = 1 if request.admission_type == "Urgent" else 0
    
    # Handle medication
    for med in ['Ibuprofen', 'Lipitor', 'Paracetamol', 'Penicillin']:
        input_data[f'Medication_{med}'] = 1 if request.medication == med else 0
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=FEATURES, fill_value=0)
    
    return input_df

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        print("🚀 API started successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model on startup: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Healthcare Predictor API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        model_status = "loaded" if model is not None else "not loaded"
        return {
            "status": "healthy",
            "model_status": model_status,
            "features_count": len(FEATURES)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction based on patient data"""
    try:
        # Load model if not already loaded
        current_model = load_model()
        
        # Prepare input data
        input_df = prepare_prediction_data(request)
        
        # Make prediction
        prediction = current_model.predict(input_df)
        
        # Return response
        return PredictionResponse(
            success=True,
            prediction=str(prediction[0]),
            inputs=request.dict()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/predict")
async def predict_get(
    age: int = 30,
    billing_amount: float = 1000.0,
    room_number: int = 101,
    gender: str = "Female",
    blood_type: str = "A+",
    medical_condition: str = "Arthritis",
    admission_type: str = "Elective",
    medication: str = "Aspirin"
):
    """GET version of predict for easy testing"""
    request = PredictionRequest(
        age=age,
        billing_amount=billing_amount,
        room_number=room_number,
        gender=gender,
        blood_type=blood_type,
        medical_condition=medical_condition,
        admission_type=admission_type,
        medication=medication
    )
    return await predict(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
