#!/usr/bin/env python3
"""
Backend API Testing Suite for Salary Prediction & Employee Management System
Tests authentication system, ML models, prediction endpoints, and API functionality
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
import sys
import uuid

# Backend URL from environment
BACKEND_URL = "https://63ccffd6-4d43-42a8-8a29-ca8b4efba19e.preview.emergentagent.com/api"

class BackendTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.auth_tokens = {}  # Store tokens for different users
        
    def log_test(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "response_data": response_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if details:
            print(f"    Details: {details}")
        if not success and response_data:
            print(f"    Response: {response_data}")
        print()

    def test_api_health(self):
        """Test basic API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                self.log_test("API Health Check", True, f"API is accessible, message: {data.get('message', 'N/A')}")
                return True
            else:
                self.log_test("API Health Check", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_test("API Health Check", False, f"Connection error: {str(e)}")
            return False

    def test_user_registration(self):
        """Test user registration with various scenarios including new skills and nomEntreprise fields"""
        test_cases = [
            {
                "name": "Valid Admin Registration with Skills",
                "data": {
                    "username": f"admin_user_{uuid.uuid4().hex[:8]}",
                    "email": f"admin_{uuid.uuid4().hex[:8]}@techcorp.com",
                    "password": "securepass123",
                    "role": "admin",
                    "name": "Sarah Johnson",
                    "skills": ["Python", "Machine Learning", "AWS", "Leadership"],
                    "nomEntreprise": "TechCorp Solutions"
                },
                "should_succeed": True
            },
            {
                "name": "Valid Employee Registration with Skills", 
                "data": {
                    "username": f"employee_user_{uuid.uuid4().hex[:8]}",
                    "email": f"employee_{uuid.uuid4().hex[:8]}@datatech.com",
                    "password": "password123",
                    "role": "employee",
                    "name": "Michael Chen",
                    "skills": ["JavaScript", "React", "Node.js", "SQL"],
                    "nomEntreprise": "DataTech Inc"
                },
                "should_succeed": True
            },
            {
                "name": "Valid Financial Analyst Registration with Skills",
                "data": {
                    "username": f"analyst_user_{uuid.uuid4().hex[:8]}",
                    "email": f"analyst_{uuid.uuid4().hex[:8]}@financeai.com", 
                    "password": "analyst123",
                    "role": "financial_analyst",
                    "name": "Emma Rodriguez",
                    "skills": ["Excel", "Tableau", "Python", "Statistics", "Power BI"],
                    "nomEntreprise": "FinanceAI Corp"
                },
                "should_succeed": True
            },
            {
                "name": "Registration without nomEntreprise (should fail)",
                "data": {
                    "username": f"no_company_{uuid.uuid4().hex[:8]}",
                    "email": f"nocompany_{uuid.uuid4().hex[:8]}@test.com",
                    "password": "password123",
                    "role": "employee",
                    "name": "Test User",
                    "skills": ["Python"]
                    # Missing nomEntreprise
                },
                "should_succeed": False
            },
            {
                "name": "Password Too Short (5 chars)",
                "data": {
                    "username": f"short_pass_{uuid.uuid4().hex[:8]}",
                    "email": f"shortpass_{uuid.uuid4().hex[:8]}@company.com",
                    "password": "12345",  # Only 5 characters
                    "role": "employee",
                    "name": "Short Pass User"
                },
                "should_succeed": False
            },
            {
                "name": "Invalid Email Format",
                "data": {
                    "username": f"invalid_email_{uuid.uuid4().hex[:8]}",
                    "email": "invalid-email-format",
                    "password": "password123",
                    "role": "employee", 
                    "name": "Invalid Email User"
                },
                "should_succeed": False
            },
            {
                "name": "Invalid Role",
                "data": {
                    "username": f"invalid_role_{uuid.uuid4().hex[:8]}",
                    "email": f"invalidrole_{uuid.uuid4().hex[:8]}@company.com",
                    "password": "password123",
                    "role": "invalid_role",
                    "name": "Invalid Role User"
                },
                "should_succeed": False
            }
        ]
        
        all_passed = True
        registered_users = []
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/auth/register",
                    json=test_case["data"],
                    headers={"Content-Type": "application/json"}
                )
                
                if test_case["should_succeed"]:
                    if response.status_code == 200:
                        data = response.json()
                        required_fields = ["id", "username", "email", "role", "name", "created_at", "is_active"]
                        missing_fields = [field for field in required_fields if field not in data]
                        
                        if missing_fields:
                            self.log_test(f"Registration - {test_case['name']}", False, f"Missing response fields: {missing_fields}", data)
                            all_passed = False
                        else:
                            # Verify skills and nomEntreprise are properly stored
                            expected_skills = test_case["data"].get("skills", [])
                            expected_company = test_case["data"].get("nomEntreprise", "")
                            
                            if "skills" in data and "nomEntreprise" in data:
                                if data["skills"] == expected_skills and data["nomEntreprise"] == expected_company:
                                    self.log_test(f"Registration - {test_case['name']}", True, f"User registered successfully: {data['email']}, Role: {data['role']}, Skills: {len(data['skills'])}, Company: {data['nomEntreprise']}")
                                else:
                                    self.log_test(f"Registration - {test_case['name']}", False, f"Skills or company mismatch. Expected skills: {expected_skills}, Got: {data.get('skills')}. Expected company: {expected_company}, Got: {data.get('nomEntreprise')}", data)
                                    all_passed = False
                            else:
                                self.log_test(f"Registration - {test_case['name']}", True, f"User registered successfully: {data['email']}, Role: {data['role']}")
                            registered_users.append(test_case["data"])
                    else:
                        self.log_test(f"Registration - {test_case['name']}", False, f"HTTP {response.status_code}", response.text)
                        all_passed = False
                else:
                    if response.status_code in [400, 422]:
                        self.log_test(f"Registration - {test_case['name']}", True, f"Correctly rejected with HTTP {response.status_code}")
                    else:
                        self.log_test(f"Registration - {test_case['name']}", False, f"Should have failed but got HTTP {response.status_code}", response.text)
                        all_passed = False
                        
            except Exception as e:
                self.log_test(f"Registration - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        # Test duplicate email registration
        if registered_users:
            duplicate_user = registered_users[0].copy()
            duplicate_user["username"] = f"duplicate_{uuid.uuid4().hex[:8]}"
            
            try:
                response = self.session.post(
                    f"{self.base_url}/auth/register",
                    json=duplicate_user,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 400:
                    response_data = response.json()
                    if "already exists" in response_data.get("detail", "").lower():
                        self.log_test("Registration - Duplicate Email Validation", True, "Correctly rejected duplicate email")
                    else:
                        self.log_test("Registration - Duplicate Email Validation", False, f"Wrong error message: {response_data.get('detail')}", response_data)
                        all_passed = False
                else:
                    self.log_test("Registration - Duplicate Email Validation", False, f"Should have failed with 400 but got HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test("Registration - Duplicate Email Validation", False, f"Error: {str(e)}")
                all_passed = False
        
        # Store registered users for login tests
        self.registered_users = registered_users
        return all_passed

    def test_user_login(self):
        """Test user login with various scenarios"""
        if not hasattr(self, 'registered_users') or not self.registered_users:
            self.log_test("Login Tests", False, "No registered users available for login testing")
            return False
        
        all_passed = True
        
        # Test valid logins for each registered user
        for user_data in self.registered_users:
            try:
                login_data = {
                    "email": user_data["email"],
                    "password": user_data["password"]
                }
                
                response = self.session.post(
                    f"{self.base_url}/auth/login",
                    json=login_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["access_token", "token_type", "user"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        self.log_test(f"Login - {user_data['role']} user", False, f"Missing response fields: {missing_fields}", data)
                        all_passed = False
                    else:
                        # Validate token format
                        token = data["access_token"]
                        if not token or len(token) < 10:
                            self.log_test(f"Login - {user_data['role']} user", False, "Invalid JWT token format", data)
                            all_passed = False
                        else:
                            # Store token for protected route tests
                            self.auth_tokens[user_data["role"]] = token
                            self.log_test(f"Login - {user_data['role']} user", True, f"Login successful, token received, user: {data['user']['email']}")
                else:
                    self.log_test(f"Login - {user_data['role']} user", False, f"HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Login - {user_data['role']} user", False, f"Error: {str(e)}")
                all_passed = False
        
        # Test invalid login scenarios
        invalid_login_cases = [
            {
                "name": "Wrong Password",
                "data": {
                    "email": self.registered_users[0]["email"],
                    "password": "wrongpassword123"
                }
            },
            {
                "name": "Non-existent Email",
                "data": {
                    "email": f"nonexistent_{uuid.uuid4().hex[:8]}@company.com",
                    "password": "password123"
                }
            },
            {
                "name": "Invalid Email Format",
                "data": {
                    "email": "invalid-email",
                    "password": "password123"
                }
            }
        ]
        
        for test_case in invalid_login_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/auth/login",
                    json=test_case["data"],
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 401:
                    self.log_test(f"Login - {test_case['name']}", True, "Correctly rejected invalid credentials")
                elif response.status_code == 422:
                    self.log_test(f"Login - {test_case['name']}", True, "Correctly rejected invalid format")
                else:
                    self.log_test(f"Login - {test_case['name']}", False, f"Should have failed but got HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Login - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_protected_routes(self):
        """Test protected routes and role-based access control"""
        if not self.auth_tokens:
            self.log_test("Protected Routes Tests", False, "No auth tokens available for testing")
            return False
        
        all_passed = True
        
        # Test /api/auth/me endpoint with valid tokens
        for role, token in self.auth_tokens.items():
            try:
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
                
                response = self.session.get(f"{self.base_url}/auth/me", headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["id", "username", "email", "role", "name", "created_at", "is_active"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        self.log_test(f"Protected Route /auth/me - {role}", False, f"Missing response fields: {missing_fields}", data)
                        all_passed = False
                    elif data["role"] != role:
                        self.log_test(f"Protected Route /auth/me - {role}", False, f"Role mismatch: expected {role}, got {data['role']}", data)
                        all_passed = False
                    else:
                        self.log_test(f"Protected Route /auth/me - {role}", True, f"Successfully retrieved user info: {data['email']}")
                else:
                    self.log_test(f"Protected Route /auth/me - {role}", False, f"HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Protected Route /auth/me - {role}", False, f"Error: {str(e)}")
                all_passed = False
        
        # Test /api/models-comparison with role-based access
        role_access_tests = [
            {"role": "admin", "should_have_access": True},
            {"role": "financial_analyst", "should_have_access": True},
            {"role": "employee", "should_have_access": False}
        ]
        
        for test_case in role_access_tests:
            role = test_case["role"]
            should_have_access = test_case["should_have_access"]
            
            if role not in self.auth_tokens:
                continue
                
            try:
                headers = {
                    "Authorization": f"Bearer {self.auth_tokens[role]}",
                    "Content-Type": "application/json"
                }
                
                response = self.session.get(f"{self.base_url}/models-comparison", headers=headers)
                
                if should_have_access:
                    if response.status_code == 200:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
                            self.log_test(f"Role-based Access /models-comparison - {role}", True, f"Access granted, {len(data)} models returned")
                        else:
                            self.log_test(f"Role-based Access /models-comparison - {role}", False, "Access granted but invalid response", data)
                            all_passed = False
                    else:
                        self.log_test(f"Role-based Access /models-comparison - {role}", False, f"Should have access but got HTTP {response.status_code}", response.text)
                        all_passed = False
                else:
                    if response.status_code == 403:
                        self.log_test(f"Role-based Access /models-comparison - {role}", True, "Access correctly denied")
                    else:
                        self.log_test(f"Role-based Access /models-comparison - {role}", False, f"Should be denied but got HTTP {response.status_code}", response.text)
                        all_passed = False
                        
            except Exception as e:
                self.log_test(f"Role-based Access /models-comparison - {role}", False, f"Error: {str(e)}")
                all_passed = False
        
        # Test invalid token scenarios
        invalid_token_tests = [
            {"name": "No Authorization Header", "headers": {"Content-Type": "application/json"}},
            {"name": "Invalid Token Format", "headers": {"Authorization": "Bearer invalid_token", "Content-Type": "application/json"}},
            {"name": "Malformed Authorization Header", "headers": {"Authorization": "InvalidFormat", "Content-Type": "application/json"}}
        ]
        
        for test_case in invalid_token_tests:
            try:
                response = self.session.get(f"{self.base_url}/auth/me", headers=test_case["headers"])
                
                if response.status_code == 401:
                    self.log_test(f"Invalid Token - {test_case['name']}", True, "Correctly rejected invalid token")
                else:
                    self.log_test(f"Invalid Token - {test_case['name']}", False, f"Should have failed with 401 but got HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Invalid Token - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_available_options_api(self):
        """Test /api/available-options endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/available-options")
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = [
                    "experience_levels", "employment_types", "job_titles", 
                    "countries", "company_sizes", "experience_level_descriptions",
                    "employment_type_descriptions", "company_size_descriptions"
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Available Options API", False, f"Missing fields: {missing_fields}", data)
                    return False
                
                # Validate specific content
                if len(data["experience_levels"]) != 4 or "EN" not in data["experience_levels"]:
                    self.log_test("Available Options API", False, "Invalid experience levels", data["experience_levels"])
                    return False
                
                if len(data["job_titles"]) < 5:
                    self.log_test("Available Options API", False, "Insufficient job titles", data["job_titles"])
                    return False
                
                self.log_test("Available Options API", True, f"All required options provided. Job titles: {len(data['job_titles'])}, Countries: {len(data['countries'])}")
                return True
            else:
                self.log_test("Available Options API", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_test("Available Options API", False, f"Error: {str(e)}")
            return False

    def test_models_comparison_api(self):
        """Test /api/models-comparison endpoint"""
        # Use admin token for authentication
        headers = {"Content-Type": "application/json"}
        if "admin" in self.auth_tokens:
            headers["Authorization"] = f"Bearer {self.auth_tokens['admin']}"
        
        try:
            response = self.session.get(f"{self.base_url}/models-comparison", headers=headers)
            if response.status_code == 200:
                data = response.json()
                
                if not isinstance(data, list) or len(data) != 7:
                    self.log_test("Models Comparison API", False, f"Expected 7 models, got {len(data) if isinstance(data, list) else 'non-list'}", data)
                    return False
                
                # Check required fields for each model
                required_fields = ["model_name", "r2_score", "mae", "rmse", "mse", "training_time"]
                expected_models = [
                    "Linear Regression", "Ridge Regression", "Random Forest", 
                    "Decision Tree", "Gradient Boosting", "AdaBoost", "XGBoost"
                ]
                
                found_models = []
                best_model = None
                best_r2 = -1
                
                for model in data:
                    missing_fields = [field for field in required_fields if field not in model]
                    if missing_fields:
                        self.log_test("Models Comparison API", False, f"Model {model.get('model_name', 'Unknown')} missing fields: {missing_fields}", model)
                        return False
                    
                    found_models.append(model["model_name"])
                    
                    # Track best model
                    if model["r2_score"] > best_r2:
                        best_r2 = model["r2_score"]
                        best_model = model["model_name"]
                    
                    # Validate metric ranges (allow negative R² for Ridge regression)
                    if model["r2_score"] > 1:
                        self.log_test("Models Comparison API", False, f"Invalid R² score for {model['model_name']}: {model['r2_score']}", model)
                        return False
                
                # Check if all expected models are present
                missing_models = [model for model in expected_models if model not in found_models]
                if missing_models:
                    self.log_test("Models Comparison API", False, f"Missing models: {missing_models}", found_models)
                    return False
                
                self.log_test("Models Comparison API", True, f"All 7 models present. Best model: {best_model} (R²: {best_r2:.4f})")
                return True, best_model, best_r2
            else:
                self.log_test("Models Comparison API", False, f"HTTP {response.status_code}", response.text)
                return False
        except Exception as e:
            self.log_test("Models Comparison API", False, f"Error: {str(e)}")
            return False

    def test_salary_prediction_api(self):
        """Test /api/predict-salary endpoint with various scenarios"""
        # Use admin token for authentication
        headers = {"Content-Type": "application/json"}
        if "admin" in self.auth_tokens:
            headers["Authorization"] = f"Bearer {self.auth_tokens['admin']}"
        
        test_cases = [
            {
                "name": "Senior Data Scientist, US, Large Company, Full-time",
                "data": {
                    "work_year": 2024,
                    "experience_level": "SE",
                    "employment_type": "FT",
                    "job_title": "Data Scientist",
                    "employee_residence": "US",
                    "remote_ratio": 0,
                    "company_location": "US",
                    "company_size": "L",
                    "skills": ["Python", "Machine Learning", "SQL", "Statistics"],
                    "nomEntreprise": "Google"
                },
                "expected_salary_range": (120000, 300000)
            },
            {
                "name": "Entry-level Software Engineer, IN, Small Company, Remote",
                "data": {
                    "work_year": 2024,
                    "experience_level": "EN",
                    "employment_type": "FT",
                    "job_title": "Software Engineer",
                    "employee_residence": "IN",
                    "remote_ratio": 100,
                    "company_location": "IN",
                    "company_size": "S",
                    "skills": ["JavaScript", "React", "Node.js"],
                    "nomEntreprise": "TechCorp"
                },
                "expected_salary_range": (25000, 80000)
            },
            {
                "name": "Executive Product Manager, CH, Medium Company, Hybrid",
                "data": {
                    "work_year": 2024,
                    "experience_level": "EX",
                    "employment_type": "FT",
                    "job_title": "Product Manager",
                    "employee_residence": "CH",
                    "remote_ratio": 50,
                    "company_location": "CH",
                    "company_size": "M",
                    "skills": ["Leadership", "Strategy", "Analytics"],
                    "nomEntreprise": "Microsoft"
                },
                "expected_salary_range": (150000, 350000)
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/predict-salary",
                    json=test_case["data"],
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check required response fields
                    required_fields = ["predicted_salary_usd", "model_name", "confidence_score", "input_data"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        self.log_test(f"Salary Prediction - {test_case['name']}", False, f"Missing response fields: {missing_fields}", data)
                        all_passed = False
                        continue
                    
                    predicted_salary = data["predicted_salary_usd"]
                    model_name = data["model_name"]
                    confidence = data["confidence_score"]
                    
                    # Validate salary range
                    min_salary, max_salary = test_case["expected_salary_range"]
                    salary_in_range = min_salary <= predicted_salary <= max_salary
                    
                    # Validate confidence score
                    valid_confidence = 0 <= confidence <= 1
                    
                    if salary_in_range and valid_confidence:
                        self.log_test(
                            f"Salary Prediction - {test_case['name']}", 
                            True, 
                            f"Predicted: ${predicted_salary:,.2f}, Model: {model_name}, Confidence: {confidence:.4f}"
                        )
                    else:
                        issues = []
                        if not salary_in_range:
                            issues.append(f"Salary ${predicted_salary:,.2f} outside expected range ${min_salary:,}-${max_salary:,}")
                        if not valid_confidence:
                            issues.append(f"Invalid confidence score: {confidence}")
                        
                        self.log_test(
                            f"Salary Prediction - {test_case['name']}", 
                            False, 
                            "; ".join(issues), 
                            data
                        )
                        all_passed = False
                else:
                    self.log_test(f"Salary Prediction - {test_case['name']}", False, f"HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Salary Prediction - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Use admin token for authentication
        headers = {"Content-Type": "application/json"}
        if "admin" in self.auth_tokens:
            headers["Authorization"] = f"Bearer {self.auth_tokens['admin']}"
        
        edge_cases = [
            {
                "name": "Invalid Experience Level",
                "data": {
                    "work_year": 2024,
                    "experience_level": "INVALID",
                    "employment_type": "FT",
                    "job_title": "Data Scientist",
                    "employee_residence": "US",
                    "remote_ratio": 0,
                    "company_location": "US",
                    "company_size": "L",
                    "skills": ["Python", "SQL"],
                    "nomEntreprise": "TechCorp"
                },
                "should_fail": False  # Should handle gracefully
            },
            {
                "name": "Missing Required Field",
                "data": {
                    "work_year": 2024,
                    "experience_level": "SE",
                    "employment_type": "FT",
                    # Missing job_title
                    "employee_residence": "US",
                    "remote_ratio": 0,
                    "company_location": "US",
                    "company_size": "L",
                    "skills": ["Python"],
                    "nomEntreprise": "TechCorp"
                },
                "should_fail": True
            },
            {
                "name": "Invalid Work Year",
                "data": {
                    "work_year": 2030,  # Future year
                    "experience_level": "SE",
                    "employment_type": "FT",
                    "job_title": "Data Scientist",
                    "employee_residence": "US",
                    "remote_ratio": 0,
                    "company_location": "US",
                    "company_size": "L",
                    "skills": ["Python"],
                    "nomEntreprise": "TechCorp"
                },
                "should_fail": True
            },
            {
                "name": "Unseen Job Title",
                "data": {
                    "work_year": 2024,
                    "experience_level": "SE",
                    "employment_type": "FT",
                    "job_title": "Quantum Computing Specialist",  # Not in training data
                    "employee_residence": "US",
                    "remote_ratio": 0,
                    "company_location": "US",
                    "company_size": "L",
                    "skills": ["Quantum Computing", "Physics"],
                    "nomEntreprise": "IBM"
                },
                "should_fail": False  # Should handle gracefully
            }
        ]
        
        all_passed = True
        
        for test_case in edge_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/predict-salary",
                    json=test_case["data"],
                    headers=headers
                )
                
                if test_case["should_fail"]:
                    if response.status_code == 422 or response.status_code == 400:
                        self.log_test(f"Edge Case - {test_case['name']}", True, f"Correctly rejected with HTTP {response.status_code}")
                    else:
                        self.log_test(f"Edge Case - {test_case['name']}", False, f"Should have failed but got HTTP {response.status_code}", response.text)
                        all_passed = False
                else:
                    if response.status_code == 200:
                        data = response.json()
                        if "predicted_salary_usd" in data:
                            self.log_test(f"Edge Case - {test_case['name']}", True, f"Handled gracefully, predicted: ${data['predicted_salary_usd']:,.2f}")
                        else:
                            self.log_test(f"Edge Case - {test_case['name']}", False, "Missing prediction in response", data)
                            all_passed = False
                    else:
                        self.log_test(f"Edge Case - {test_case['name']}", False, f"Should have succeeded but got HTTP {response.status_code}", response.text)
                        all_passed = False
                        
            except Exception as e:
                self.log_test(f"Edge Case - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_skills_management_system(self):
        """Test skills management endpoints"""
        all_passed = True
        
        # Test 1: Get all available skills
        try:
            response = self.session.get(f"{self.base_url}/skills/all")
            if response.status_code == 200:
                data = response.json()
                if "skills" in data and isinstance(data["skills"], list):
                    default_skills_count = len(data["skills"])
                    self.log_test("Skills Management - Get All Skills", True, f"Retrieved {default_skills_count} skills successfully")
                else:
                    self.log_test("Skills Management - Get All Skills", False, "Invalid response format", data)
                    all_passed = False
            else:
                self.log_test("Skills Management - Get All Skills", False, f"HTTP {response.status_code}", response.text)
                all_passed = False
        except Exception as e:
            self.log_test("Skills Management - Get All Skills", False, f"Error: {str(e)}")
            all_passed = False
        
        # Test 2: Add custom skill (requires authentication)
        if "admin" in self.auth_tokens:
            headers = {
                "Authorization": f"Bearer {self.auth_tokens['admin']}",
                "Content-Type": "application/json"
            }
            
            custom_skill = f"Custom_Skill_{uuid.uuid4().hex[:8]}"
            
            try:
                response = self.session.post(
                    f"{self.base_url}/skills/add",
                    params={"skill_name": custom_skill},
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "message" in data and "skill" in data:
                        self.log_test("Skills Management - Add Custom Skill", True, f"Added custom skill: {custom_skill}")
                        
                        # Verify the skill was added by getting all skills again
                        response = self.session.get(f"{self.base_url}/skills/all")
                        if response.status_code == 200:
                            updated_data = response.json()
                            if custom_skill in updated_data.get("skills", []):
                                self.log_test("Skills Management - Verify Custom Skill Added", True, f"Custom skill {custom_skill} found in skills list")
                            else:
                                self.log_test("Skills Management - Verify Custom Skill Added", False, f"Custom skill {custom_skill} not found in updated skills list")
                                all_passed = False
                    else:
                        self.log_test("Skills Management - Add Custom Skill", False, "Invalid response format", data)
                        all_passed = False
                else:
                    self.log_test("Skills Management - Add Custom Skill", False, f"HTTP {response.status_code}", response.text)
                    all_passed = False
            except Exception as e:
                self.log_test("Skills Management - Add Custom Skill", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_enhanced_salary_prediction_with_skills(self):
        """Test enhanced salary prediction with skills and nomEntreprise"""
        if "admin" not in self.auth_tokens:
            self.log_test("Enhanced Salary Prediction Tests", False, "No admin token available")
            return False
        
        headers = {
            "Authorization": f"Bearer {self.auth_tokens['admin']}",
            "Content-Type": "application/json"
        }
        
        test_cases = [
            {
                "name": "High-Value Skills Impact (ML Engineer with ML skills)",
                "data": {
                    "work_year": 2024,
                    "experience_level": "SE",
                    "employment_type": "FT",
                    "job_title": "Machine Learning Engineer",
                    "employee_residence": "US",
                    "remote_ratio": 0,
                    "company_location": "US",
                    "company_size": "L",
                    "skills": ["Machine Learning", "Deep Learning", "TensorFlow", "Python", "AWS"],
                    "nomEntreprise": "Google"
                },
                "expected_salary_range": (180000, 280000)
            },
            {
                "name": "Standard Skills Impact (Data Analyst with basic skills)",
                "data": {
                    "work_year": 2024,
                    "experience_level": "MI",
                    "employment_type": "FT",
                    "job_title": "Data Analyst",
                    "employee_residence": "US",
                    "remote_ratio": 50,
                    "company_location": "US",
                    "company_size": "M",
                    "skills": ["Excel", "PowerBI", "SQL"],
                    "nomEntreprise": "TechCorp"
                },
                "expected_salary_range": (70000, 120000)
            },
            {
                "name": "FAANG Company Premium (Software Engineer at Meta)",
                "data": {
                    "work_year": 2024,
                    "experience_level": "SE",
                    "employment_type": "FT",
                    "job_title": "Software Engineer",
                    "employee_residence": "US",
                    "remote_ratio": 25,
                    "company_location": "US",
                    "company_size": "L",
                    "skills": ["JavaScript", "React", "Node.js", "Python"],
                    "nomEntreprise": "Meta"
                },
                "expected_salary_range": (160000, 240000)
            },
            {
                "name": "Skills vs No Skills Comparison (same profile, no skills)",
                "data": {
                    "work_year": 2024,
                    "experience_level": "MI",
                    "employment_type": "FT",
                    "job_title": "Data Scientist",
                    "employee_residence": "US",
                    "remote_ratio": 0,
                    "company_location": "US",
                    "company_size": "M",
                    "skills": [],  # No skills
                    "nomEntreprise": "Startup Inc"
                },
                "expected_salary_range": (80000, 140000)
            }
        ]
        
        all_passed = True
        predictions = []
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/predict-salary",
                    json=test_case["data"],
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check required response fields
                    required_fields = ["predicted_salary_usd", "model_name", "confidence_score", "input_data"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        self.log_test(f"Enhanced Prediction - {test_case['name']}", False, f"Missing response fields: {missing_fields}", data)
                        all_passed = False
                        continue
                    
                    predicted_salary = data["predicted_salary_usd"]
                    model_name = data["model_name"]
                    confidence = data["confidence_score"]
                    
                    # Verify skills and nomEntreprise are in input_data
                    input_data = data["input_data"]
                    if "skills" not in input_data or "nomEntreprise" not in input_data:
                        self.log_test(f"Enhanced Prediction - {test_case['name']}", False, "Skills or nomEntreprise missing from input_data", data)
                        all_passed = False
                        continue
                    
                    # Validate salary range
                    min_salary, max_salary = test_case["expected_salary_range"]
                    salary_in_range = min_salary <= predicted_salary <= max_salary
                    
                    # Store prediction for comparison
                    predictions.append({
                        "name": test_case["name"],
                        "salary": predicted_salary,
                        "skills_count": len(test_case["data"]["skills"]),
                        "company": test_case["data"]["nomEntreprise"]
                    })
                    
                    if salary_in_range:
                        self.log_test(
                            f"Enhanced Prediction - {test_case['name']}", 
                            True, 
                            f"Predicted: ${predicted_salary:,.2f}, Skills: {len(test_case['data']['skills'])}, Company: {test_case['data']['nomEntreprise']}, Model: {model_name}"
                        )
                    else:
                        self.log_test(
                            f"Enhanced Prediction - {test_case['name']}", 
                            False, 
                            f"Salary ${predicted_salary:,.2f} outside expected range ${min_salary:,}-${max_salary:,}",
                            data
                        )
                        all_passed = False
                else:
                    self.log_test(f"Enhanced Prediction - {test_case['name']}", False, f"HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Enhanced Prediction - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        # Verify skills impact: High-value skills should result in higher salaries
        if len(predictions) >= 2:
            high_skill_pred = next((p for p in predictions if "High-Value Skills" in p["name"]), None)
            standard_skill_pred = next((p for p in predictions if "Standard Skills" in p["name"]), None)
            
            if high_skill_pred and standard_skill_pred:
                if high_skill_pred["salary"] > standard_skill_pred["salary"]:
                    self.log_test("Skills Impact Verification", True, f"High-value skills resulted in higher salary: ${high_skill_pred['salary']:,.2f} vs ${standard_skill_pred['salary']:,.2f}")
                else:
                    self.log_test("Skills Impact Verification", False, f"High-value skills did not result in higher salary: ${high_skill_pred['salary']:,.2f} vs ${standard_skill_pred['salary']:,.2f}")
                    all_passed = False
        
        return all_passed

    def test_advanced_analytics_endpoints(self):
        """Test all 6 analytics endpoints for financial analysts"""
        if "financial_analyst" not in self.auth_tokens:
            self.log_test("Analytics Endpoints Tests", False, "No financial_analyst token available")
            return False
        
        headers = {
            "Authorization": f"Bearer {self.auth_tokens['financial_analyst']}",
            "Content-Type": "application/json"
        }
        
        analytics_endpoints = [
            {
                "endpoint": "/analytics/salary-trends",
                "name": "Salary Trends",
                "expected_fields": ["trends"]
            },
            {
                "endpoint": "/analytics/company-summaries", 
                "name": "Company Summaries",
                "expected_fields": ["summaries"]
            },
            {
                "endpoint": "/analytics/correlation-heatmap",
                "name": "Correlation Heatmap",
                "expected_fields": ["correlation_matrix", "variables"]
            },
            {
                "endpoint": "/analytics/top-rankings",
                "name": "Top Rankings",
                "expected_fields": ["top_jobs", "top_companies", "top_skills"]
            },
            {
                "endpoint": "/analytics/annual-summary",
                "name": "Annual Summary", 
                "expected_fields": ["annual_summaries"]
            },
            {
                "endpoint": "/analytics/salary-distribution",
                "name": "Salary Distribution",
                "expected_fields": ["distribution_by_company_size", "distribution_by_experience", "salary_histogram"]
            }
        ]
        
        all_passed = True
        
        for endpoint_test in analytics_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint_test['endpoint']}", headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for expected fields
                    missing_fields = [field for field in endpoint_test["expected_fields"] if field not in data]
                    
                    if missing_fields:
                        self.log_test(f"Analytics - {endpoint_test['name']}", False, f"Missing fields: {missing_fields}", data)
                        all_passed = False
                    else:
                        # Validate data structure based on endpoint
                        if endpoint_test["endpoint"] == "/analytics/salary-trends":
                            trends = data.get("trends", [])
                            if isinstance(trends, list) and len(trends) > 0:
                                sample_trend = trends[0]
                                required_trend_fields = ["job_title", "company", "avg_salary", "count"]
                                if all(field in sample_trend for field in required_trend_fields):
                                    self.log_test(f"Analytics - {endpoint_test['name']}", True, f"Retrieved {len(trends)} salary trends")
                                else:
                                    self.log_test(f"Analytics - {endpoint_test['name']}", False, f"Invalid trend structure: {sample_trend}")
                                    all_passed = False
                            else:
                                self.log_test(f"Analytics - {endpoint_test['name']}", False, "No trends data returned")
                                all_passed = False
                        
                        elif endpoint_test["endpoint"] == "/analytics/company-summaries":
                            summaries = data.get("summaries", [])
                            if isinstance(summaries, list) and len(summaries) > 0:
                                sample_summary = summaries[0]
                                required_summary_fields = ["company", "total_employees", "total_salary_cost", "avg_salary", "monthly_cost", "annual_cost"]
                                if all(field in sample_summary for field in required_summary_fields):
                                    self.log_test(f"Analytics - {endpoint_test['name']}", True, f"Retrieved {len(summaries)} company summaries")
                                else:
                                    self.log_test(f"Analytics - {endpoint_test['name']}", False, f"Invalid summary structure: {sample_summary}")
                                    all_passed = False
                            else:
                                self.log_test(f"Analytics - {endpoint_test['name']}", False, "No summaries data returned")
                                all_passed = False
                        
                        elif endpoint_test["endpoint"] == "/analytics/top-rankings":
                            rankings = data
                            required_ranking_fields = ["top_jobs", "top_companies", "top_skills"]
                            if all(field in rankings and isinstance(rankings[field], list) for field in required_ranking_fields):
                                jobs_count = len(rankings["top_jobs"])
                                companies_count = len(rankings["top_companies"])
                                skills_count = len(rankings["top_skills"])
                                self.log_test(f"Analytics - {endpoint_test['name']}", True, f"Retrieved rankings - Jobs: {jobs_count}, Companies: {companies_count}, Skills: {skills_count}")
                            else:
                                self.log_test(f"Analytics - {endpoint_test['name']}", False, "Invalid rankings structure", rankings)
                                all_passed = False
                        
                        else:
                            # For other endpoints, just check that expected fields exist and contain data
                            data_count = sum(1 for field in endpoint_test["expected_fields"] if data.get(field))
                            self.log_test(f"Analytics - {endpoint_test['name']}", True, f"Retrieved data with {data_count}/{len(endpoint_test['expected_fields'])} expected fields")
                
                else:
                    self.log_test(f"Analytics - {endpoint_test['name']}", False, f"HTTP {response.status_code}", response.text)
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Analytics - {endpoint_test['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_role_based_analytics_access(self):
        """Test role-based access control for analytics endpoints"""
        analytics_endpoint = "/analytics/salary-trends"
        all_passed = True
        
        # Test access for different roles
        role_tests = [
            {"role": "financial_analyst", "should_have_access": True},
            {"role": "admin", "should_have_access": True},
            {"role": "employee", "should_have_access": False}
        ]
        
        for test_case in role_tests:
            role = test_case["role"]
            should_have_access = test_case["should_have_access"]
            
            if role not in self.auth_tokens:
                continue
            
            headers = {
                "Authorization": f"Bearer {self.auth_tokens[role]}",
                "Content-Type": "application/json"
            }
            
            try:
                response = self.session.get(f"{self.base_url}{analytics_endpoint}", headers=headers)
                
                if should_have_access:
                    if response.status_code == 200:
                        self.log_test(f"Analytics Access Control - {role}", True, "Access granted correctly")
                    else:
                        self.log_test(f"Analytics Access Control - {role}", False, f"Should have access but got HTTP {response.status_code}", response.text)
                        all_passed = False
                else:
                    if response.status_code == 403:
                        self.log_test(f"Analytics Access Control - {role}", True, "Access correctly denied")
                    else:
                        self.log_test(f"Analytics Access Control - {role}", False, f"Should be denied but got HTTP {response.status_code}", response.text)
                        all_passed = False
                        
            except Exception as e:
                self.log_test(f"Analytics Access Control - {role}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def run_all_tests(self):
        """Run all backend tests"""
        print("=" * 80)
        print("BACKEND API TESTING SUITE")
        print("=" * 80)
        print(f"Testing backend at: {self.base_url}")
        print()
        
        # Test 1: API Health
        if not self.test_api_health():
            print("❌ API is not accessible. Stopping tests.")
            return False
        
        # Test 2: Authentication System Tests
        print("🔐 AUTHENTICATION SYSTEM TESTS")
        print("-" * 40)
        
        # Test user registration
        registration_result = self.test_user_registration()
        
        # Test user login
        login_result = self.test_user_login()
        
        # Test protected routes and role-based access
        protected_routes_result = self.test_protected_routes()
        
        print("\n📊 ML MODELS & PREDICTION TESTS")
        print("-" * 40)
        
        # Test 3: Available Options API
        self.test_available_options_api()
        
        # Test 4: Models Comparison API
        models_result = self.test_models_comparison_api()
        
        # Test 5: Salary Prediction API
        prediction_result = self.test_salary_prediction_api()
        
        # Test 6: Edge Cases
        edge_cases_result = self.test_edge_cases()
        
        print("\n🔧 NEW FEATURES TESTING")
        print("-" * 40)
        
        # Test 7: Skills Management System
        skills_result = self.test_skills_management_system()
        
        # Test 8: Enhanced Salary Prediction with Skills
        enhanced_prediction_result = self.test_enhanced_salary_prediction_with_skills()
        
        # Test 9: Advanced Analytics Endpoints
        analytics_result = self.test_advanced_analytics_endpoints()
        
        # Test 10: Role-based Analytics Access Control
        analytics_access_result = self.test_role_based_analytics_access()
        
        # Summary
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result["success"])
        total_tests = len(self.test_results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        # Authentication Tests Summary
        auth_tests = [result for result in self.test_results if any(keyword in result["test"] for keyword in ["Registration", "Login", "Protected Route", "Role-based Access", "Invalid Token"])]
        auth_passed = sum(1 for result in auth_tests if result["success"])
        auth_total = len(auth_tests)
        
        if auth_total > 0:
            print(f"🔐 Authentication Tests: {auth_passed}/{auth_total} passed ({(auth_passed/auth_total)*100:.1f}%)")
        
        # ML Tests Summary
        ml_tests = [result for result in self.test_results if any(keyword in result["test"] for keyword in ["Models Comparison", "Salary Prediction", "Available Options", "Edge Case"])]
        ml_passed = sum(1 for result in ml_tests if result["success"])
        ml_total = len(ml_tests)
        
        if ml_total > 0:
            print(f"📊 ML & Prediction Tests: {ml_passed}/{ml_total} passed ({(ml_passed/ml_total)*100:.1f}%)")
        
        # New Features Tests Summary
        new_features_tests = [result for result in self.test_results if any(keyword in result["test"] for keyword in ["Skills Management", "Enhanced Prediction", "Analytics", "Skills Impact"])]
        new_features_passed = sum(1 for result in new_features_tests if result["success"])
        new_features_total = len(new_features_tests)
        
        if new_features_total > 0:
            print(f"🔧 New Features Tests: {new_features_passed}/{new_features_total} passed ({(new_features_passed/new_features_total)*100:.1f}%)")
        print()
        
        # Critical Issues
        critical_failures = [result for result in self.test_results if not result["success"] and 
                           any(keyword in result["test"] for keyword in ["API Health", "Registration", "Login", "Protected Route", "Models Comparison", "Salary Prediction"])]
        
        if critical_failures:
            print("🚨 CRITICAL ISSUES:")
            for failure in critical_failures:
                print(f"  - {failure['test']}: {failure['details']}")
            print()
        
        # Overall Status
        overall_success = passed_tests >= total_tests * 0.8  # 80% pass rate
        auth_success = auth_passed >= auth_total * 0.8 if auth_total > 0 else True
        
        if overall_success and auth_success:
            status = "✅ OVERALL: BACKEND TESTS PASSED"
        else:
            status = "❌ OVERALL: BACKEND TESTS FAILED"
            
        print(status)
        print("=" * 80)
        
        return overall_success and auth_success

if __name__ == "__main__":
    tester = BackendTester(BACKEND_URL)
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)