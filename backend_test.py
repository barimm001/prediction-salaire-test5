#!/usr/bin/env python3
"""
Backend API Testing Suite for Salary Prediction & Employee Management System
Tests all ML models, prediction endpoints, and API functionality
"""

import requests
import json
import time
from typing import Dict, List, Any
import sys

# Backend URL from environment
BACKEND_URL = "https://ea82922d-231d-469f-8a0e-e02931dea42c.preview.emergentagent.com/api"

class BackendTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
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
        try:
            response = self.session.get(f"{self.base_url}/models-comparison")
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
                    
                    # Validate metric ranges
                    if not (0 <= model["r2_score"] <= 1):
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
                    "company_size": "L"
                },
                "expected_salary_range": (120000, 200000)
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
                    "company_size": "S"
                },
                "expected_salary_range": (25000, 60000)
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
                    "company_size": "M"
                },
                "expected_salary_range": (150000, 250000)
            }
        ]
        
        all_passed = True
        
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/predict-salary",
                    json=test_case["data"],
                    headers={"Content-Type": "application/json"}
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
                    "company_size": "L"
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
                    "company_size": "L"
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
                    "company_size": "L"
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
                    "company_size": "L"
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
                    headers={"Content-Type": "application/json"}
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
        
        # Test 2: Available Options API
        self.test_available_options_api()
        
        # Test 3: Models Comparison API
        models_result = self.test_models_comparison_api()
        
        # Test 4: Salary Prediction API
        prediction_result = self.test_salary_prediction_api()
        
        # Test 5: Edge Cases
        edge_cases_result = self.test_edge_cases()
        
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
        
        # Critical Issues
        critical_failures = [result for result in self.test_results if not result["success"] and 
                           any(keyword in result["test"] for keyword in ["API Health", "Models Comparison", "Salary Prediction"])]
        
        if critical_failures:
            print("🚨 CRITICAL ISSUES:")
            for failure in critical_failures:
                print(f"  - {failure['test']}: {failure['details']}")
            print()
        
        # Overall Status
        overall_success = passed_tests >= total_tests * 0.8  # 80% pass rate
        status = "✅ OVERALL: BACKEND TESTS PASSED" if overall_success else "❌ OVERALL: BACKEND TESTS FAILED"
        print(status)
        print("=" * 80)
        
        return overall_success

if __name__ == "__main__":
    tester = BackendTester(BACKEND_URL)
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)