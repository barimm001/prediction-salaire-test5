from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Salary Prediction & Employee Management System", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for models and encoders
models_data = {}
encoders = {}
scaler = None

# Data Models
class SalaryPredictionInput(BaseModel):
    work_year: int = Field(..., ge=2020, le=2025, description="Year of work")
    experience_level: str = Field(..., description="EN, MI, SE, EX")
    employment_type: str = Field(..., description="FT, PT, CT, FL")
    job_title: str = Field(..., description="Job title")
    employee_residence: str = Field(..., description="Country code")
    remote_ratio: int = Field(..., ge=0, le=100, description="Remote work percentage")
    company_location: str = Field(..., description="Country code") 
    company_size: str = Field(..., description="S, M, L")

class SalaryPredictionResponse(BaseModel):
    predicted_salary_usd: float
    model_name: str
    confidence_score: float
    input_data: Dict[str, Any]
    
class ModelComparison(BaseModel):
    model_name: str
    r2_score: float
    mae: float
    rmse: float
    mse: float
    training_time: float

class Employee(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    role: str
    department: str
    hire_date: datetime = Field(default_factory=datetime.utcnow)
    salary_prediction: Optional[float] = None

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    role: str  # admin, employee, financial_analyst
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Initialize sample data and train models
async def initialize_models():
    """Initialize and train ML models with sample data"""
    global models_data, encoders, scaler
    
    # Create sample salary dataset based on the structure
    np.random.seed(42)  # For reproducible results
    n_samples = 200
    
    sample_data = {
        'work_year': np.random.choice([2023, 2024], n_samples),
        'experience_level': np.random.choice(['EN', 'MI', 'SE', 'EX'], n_samples),
        'employment_type': np.random.choice(['FT', 'PT', 'CT', 'FL'], n_samples),
        'job_title': np.random.choice([
            'Data Scientist', 'Machine Learning Engineer', 'Data Analyst', 
            'Software Engineer', 'Product Manager', 'Data Engineer',
            'Research Scientist', 'AI Engineer', 'Backend Developer',
            'Full Stack Developer'
        ], n_samples),
        'employee_residence': np.random.choice(['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'AU', 'NL', 'CH', 'ES'], n_samples),
        'remote_ratio': np.random.choice([0, 25, 50, 75, 100], n_samples),
        'company_location': np.random.choice(['US', 'CA', 'GB', 'DE', 'FR', 'IN', 'AU', 'NL', 'CH', 'ES'], n_samples),
        'company_size': np.random.choice(['S', 'M', 'L'], n_samples),
    }
    
    # Generate realistic salaries based on features
    salaries = []
    for i in range(200):
        base_salary = 70000
        
        # Experience level adjustments
        exp_multiplier = {'EN': 1.0, 'MI': 1.3, 'SE': 1.6, 'EX': 2.0}
        base_salary *= exp_multiplier[sample_data['experience_level'][i]]
        
        # Job title adjustments
        job_multiplier = {
            'Data Scientist': 1.2, 'Machine Learning Engineer': 1.3, 'Data Analyst': 0.9,
            'Software Engineer': 1.1, 'Product Manager': 1.4, 'Data Engineer': 1.15,
            'Research Scientist': 1.25, 'AI Engineer': 1.35, 'Backend Developer': 1.05,
            'Full Stack Developer': 1.1
        }
        base_salary *= job_multiplier[sample_data['job_title'][i]]
        
        # Company size adjustments
        size_multiplier = {'S': 0.9, 'M': 1.0, 'L': 1.2}
        base_salary *= size_multiplier[sample_data['company_size'][i]]
        
        # Location adjustments
        location_multiplier = {
            'US': 1.2, 'CA': 1.0, 'GB': 1.1, 'DE': 1.05, 'FR': 1.05,
            'IN': 0.4, 'AU': 1.0, 'NL': 1.1, 'CH': 1.3, 'ES': 0.8
        }
        base_salary *= location_multiplier[sample_data['company_location'][i]]
        
        # Add some randomness
        base_salary *= np.random.normal(1.0, 0.1)
        salaries.append(max(30000, int(base_salary)))
    
    sample_data['salary_in_usd'] = salaries
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Prepare features for training
    categorical_columns = ['experience_level', 'employment_type', 'job_title', 
                          'employee_residence', 'company_location', 'company_size']
    
    # Initialize encoders
    encoders = {}
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        df[col + '_encoded'] = encoders[col].fit_transform(df[col])
    
    # Prepare feature matrix
    feature_columns = ['work_year', 'remote_ratio'] + [col + '_encoded' for col in categorical_columns]
    X = df[feature_columns]
    y = df['salary_in_usd']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42)
    }
    
    # Train and evaluate models
    models_data = {}
    for name, model in models.items():
        start_time = datetime.now()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model data
        models_data[name] = {
            'model': model,
            'r2_score': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'training_time': training_time,
            'feature_columns': feature_columns
        }
    
    print("Models initialized and trained successfully!")

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Salary Prediction & Employee Management API"}

@api_router.post("/predict-salary", response_model=SalaryPredictionResponse)
async def predict_salary(input_data: SalaryPredictionInput):
    """Predict salary using the best performing model"""
    if not models_data:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    try:
        # Find the best model (highest R² score)
        best_model_name = max(models_data.keys(), key=lambda x: models_data[x]['r2_score'])
        best_model_data = models_data[best_model_name]
        
        # Prepare input for prediction
        input_dict = input_data.dict()
        
        # Encode categorical features
        categorical_columns = ['experience_level', 'employment_type', 'job_title', 
                              'employee_residence', 'company_location', 'company_size']
        
        feature_values = [input_dict['work_year'], input_dict['remote_ratio']]
        
        for col in categorical_columns:
            value = input_dict[col]
            if value in encoders[col].classes_:
                encoded_value = encoders[col].transform([value])[0]
            else:
                # Handle unseen categories by using the most common category
                encoded_value = 0
            feature_values.append(encoded_value)
        
        # Scale features
        features_scaled = scaler.transform([feature_values])
        
        # Make prediction
        prediction = best_model_data['model'].predict(features_scaled)[0]
        
        return SalaryPredictionResponse(
            predicted_salary_usd=float(prediction),
            model_name=best_model_name,
            confidence_score=best_model_data['r2_score'],
            input_data=input_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@api_router.get("/models-comparison", response_model=List[ModelComparison])
async def get_models_comparison():
    """Get comparison of all trained models"""
    if not models_data:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    comparisons = []
    for name, data in models_data.items():
        comparisons.append(ModelComparison(
            model_name=name,
            r2_score=data['r2_score'],
            mae=data['mae'],
            rmse=data['rmse'],
            mse=data['mse'],
            training_time=data['training_time']
        ))
    
    # Sort by R² score descending
    comparisons.sort(key=lambda x: x.r2_score, reverse=True)
    return comparisons

@api_router.get("/available-options")
async def get_available_options():
    """Get all available options for prediction form"""
    return {
        "experience_levels": ["EN", "MI", "SE", "EX"],
        "employment_types": ["FT", "PT", "CT", "FL"],
        "job_titles": [
            "Data Scientist", "Machine Learning Engineer", "Data Analyst",
            "Software Engineer", "Product Manager", "Data Engineer",
            "Research Scientist", "AI Engineer", "Backend Developer",
            "Full Stack Developer"
        ],
        "countries": ["US", "CA", "GB", "DE", "FR", "IN", "AU", "NL", "CH", "ES"],
        "company_sizes": ["S", "M", "L"],
        "experience_level_descriptions": {
            "EN": "Entry-level / Junior",
            "MI": "Mid-level / Intermediate", 
            "SE": "Senior-level / Expert",
            "EX": "Executive-level / Director"
        },
        "employment_type_descriptions": {
            "FT": "Full-time",
            "PT": "Part-time",
            "CT": "Contract",
            "FL": "Freelance"
        },
        "company_size_descriptions": {
            "S": "Small (less than 50 employees)",
            "M": "Medium (50-250 employees)",
            "L": "Large (250+ employees)"
        }
    }

# Employee management endpoints
@api_router.post("/employees", response_model=Employee)
async def create_employee(employee_data: Employee):
    """Create a new employee"""
    employee_dict = employee_data.dict()
    await db.employees.insert_one(employee_dict)
    return employee_data

@api_router.get("/employees", response_model=List[Employee])
async def get_employees():
    """Get all employees"""
    employees = await db.employees.find().to_list(1000)
    return [Employee(**emp) for emp in employees]

# User management endpoints
@api_router.post("/users", response_model=User)
async def create_user(user_data: User):
    """Create a new user"""
    user_dict = user_data.dict()
    await db.users.insert_one(user_dict)
    return user_data

@api_router.get("/users", response_model=List[User])
async def get_users():
    """Get all users"""
    users = await db.users.find().to_list(1000)
    return [User(**user) for user in users]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    await initialize_models()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()