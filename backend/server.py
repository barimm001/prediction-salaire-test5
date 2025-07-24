from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
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
import jwt
from passlib.context import CryptContext
import bcrypt

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

# Authentication setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Global variables for models and encoders
models_data = {}
encoders = {}
scaler = None

# Authentication Models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: str = Field(..., pattern="^(admin|employee|financial_analyst)$")
    name: str = Field(..., min_length=3, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must contain only letters, numbers, hyphens and underscores')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    name: str
    created_at: datetime
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None

# ML Models (unchanged)
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

# Employee Management Models
class EmployeeCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    role: str = Field(..., min_length=2, max_length=50)
    department: str = Field(..., min_length=2, max_length=50)
    skills: List[str] = Field(default_factory=list)
    salary_prediction: Optional[float] = None

class Employee(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    role: str
    department: str
    skills: List[str] = Field(default_factory=list)
    hire_date: datetime = Field(default_factory=datetime.utcnow)
    salary_prediction: Optional[float] = None

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, max_length=1000)
    assigned_to: str  # employee_id
    priority: str = Field(..., pattern="^(low|medium|high|urgent)$")
    due_date: datetime
    status: str = Field(default="pending", pattern="^(pending|in_progress|completed|cancelled)$")

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    assigned_to: str
    assigned_by: str  # admin/manager id
    priority: str
    due_date: datetime
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class MeetingCreate(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., max_length=1000)
    start_time: datetime
    end_time: datetime
    attendees: List[str] = Field(default_factory=list)  # employee ids
    meeting_type: str = Field(default="team", pattern="^(team|one_on_one|all_hands|client)$")

class Meeting(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    attendees: List[str] = Field(default_factory=list)
    created_by: str  # admin/manager id
    meeting_type: str = "team"
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Authentication utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await db.users.find_one({"email": token_data.email})
    if user is None:
        raise credentials_exception
    return UserResponse(**user)

async def get_current_active_user(current_user: UserResponse = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Role-based access control
def require_role(allowed_roles: List[str]):
    def role_checker(current_user: UserResponse = Depends(get_current_active_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return current_user
    return role_checker

# Initialize sample data and train models (unchanged from before)
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
    for i in range(n_samples):
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

# Authentication Routes
@api_router.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # Check if user already exists
    existing_user = await db.users.find_one({"$or": [{"email": user_data.email}, {"username": user_data.username}]})
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User with this email or username already exists"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create user document
    user_dict = {
        "id": str(uuid.uuid4()),
        "username": user_data.username,
        "email": user_data.email,
        "name": user_data.name,
        "role": user_data.role,
        "hashed_password": hashed_password,
        "is_active": True,
        "created_at": datetime.utcnow()
    }
    
    # Insert into database
    await db.users.insert_one(user_dict)
    
    # Return user data (without password)
    user_dict.pop("hashed_password")
    return UserResponse(**user_dict)

@api_router.post("/auth/login", response_model=Token)
async def login_user(user_credentials: UserLogin):
    """Login user and return JWT token"""
    user = await db.users.find_one({"email": user_credentials.email})
    
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    # Remove password from user data
    user.pop("hashed_password")
    user_response = UserResponse(**user)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )

@api_router.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserResponse = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

# Protected Routes with Role-based Access Control

# ML Prediction Routes (accessible to all authenticated users)
@api_router.get("/")
async def root():
    return {"message": "Salary Prediction & Employee Management API"}

@api_router.post("/predict-salary", response_model=SalaryPredictionResponse)
async def predict_salary(
    input_data: SalaryPredictionInput,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Predict salary using the best performing model (accessible to all authenticated users)"""
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
async def get_models_comparison(
    current_user: UserResponse = Depends(require_role(["admin", "financial_analyst"]))
):
    """Get comparison of all trained models (admin and financial analyst only)"""
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
    """Get all available options for prediction form (public access)"""
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

# Employee management endpoints (admin only)
@api_router.post("/employees", response_model=Employee)
async def create_employee(
    employee_data: EmployeeCreate,
    current_user: UserResponse = Depends(require_role(["admin"]))
):
    """Create a new employee (admin only)"""
    employee_dict = employee_data.dict()
    employee_dict["id"] = str(uuid.uuid4())
    employee_dict["hire_date"] = datetime.utcnow()
    
    await db.employees.insert_one(employee_dict)
    return Employee(**employee_dict)

@api_router.get("/employees", response_model=List[Employee])
async def get_employees(
    current_user: UserResponse = Depends(require_role(["admin"]))
):
    """Get all employees (admin only)"""
    employees = await db.employees.find().to_list(1000)
    return [Employee(**emp) for emp in employees]

@api_router.put("/employees/{employee_id}", response_model=Employee)
async def update_employee(
    employee_id: str,
    employee_data: EmployeeCreate,
    current_user: UserResponse = Depends(require_role(["admin"]))
):
    """Update an employee (admin only)"""
    employee_dict = employee_data.dict()
    employee_dict["updated_at"] = datetime.utcnow()
    
    result = await db.employees.update_one(
        {"id": employee_id},
        {"$set": employee_dict}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    updated_employee = await db.employees.find_one({"id": employee_id})
    return Employee(**updated_employee)

@api_router.delete("/employees/{employee_id}")
async def delete_employee(
    employee_id: str,
    current_user: UserResponse = Depends(require_role(["admin"]))
):
    """Delete an employee (admin only)"""
    result = await db.employees.delete_one({"id": employee_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    return {"message": "Employee deleted successfully"}

# Task management endpoints
@api_router.post("/tasks", response_model=Task)
async def create_task(
    task_data: TaskCreate,
    current_user: UserResponse = Depends(require_role(["admin"]))
):
    """Create a new task (admin only)"""
    task_dict = task_data.dict()
    task_dict["id"] = str(uuid.uuid4())
    task_dict["assigned_by"] = current_user.id
    task_dict["created_at"] = datetime.utcnow()
    task_dict["updated_at"] = datetime.utcnow()
    
    await db.tasks.insert_one(task_dict)
    return Task(**task_dict)

@api_router.get("/tasks", response_model=List[Task])
async def get_tasks(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get tasks (all for admin, only assigned for employees)"""
    if current_user.role == "admin":
        tasks = await db.tasks.find().to_list(1000)
    else:
        tasks = await db.tasks.find({"assigned_to": current_user.id}).to_list(1000)
    
    return [Task(**task) for task in tasks]

@api_router.put("/tasks/{task_id}/status")
async def update_task_status(
    task_id: str,
    status: str,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Update task status"""
    if status not in ["pending", "in_progress", "completed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    # Check if user can update this task
    task = await db.tasks.find_one({"id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if current_user.role != "admin" and task["assigned_to"] != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this task")
    
    await db.tasks.update_one(
        {"id": task_id},
        {"$set": {"status": status, "updated_at": datetime.utcnow()}}
    )
    
    return {"message": "Task status updated successfully"}

# Meeting management endpoints
@api_router.post("/meetings", response_model=Meeting)
async def create_meeting(
    meeting_data: MeetingCreate,
    current_user: UserResponse = Depends(require_role(["admin"]))
):
    """Create a new meeting (admin only)"""
    meeting_dict = meeting_data.dict()
    meeting_dict["id"] = str(uuid.uuid4())
    meeting_dict["created_by"] = current_user.id
    meeting_dict["created_at"] = datetime.utcnow()
    
    await db.meetings.insert_one(meeting_dict)
    return Meeting(**meeting_dict)

@api_router.get("/meetings", response_model=List[Meeting])
async def get_meetings(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """Get meetings (all for admin, only attended for employees)"""
    if current_user.role == "admin":
        meetings = await db.meetings.find().to_list(1000)
    else:
        meetings = await db.meetings.find({"attendees": current_user.id}).to_list(1000)
    
    return [Meeting(**meeting) for meeting in meetings]

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