import React, { useState, useEffect, createContext, useContext } from "react";
import axios from "axios";
import Select from "react-select";
import { 
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter
} from "recharts";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Auth Context
const AuthContext = createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      checkAuth();
    } else {
      setLoading(false);
    }
  }, [token]);

  const checkAuth = async () => {
    try {
      const response = await axios.get(`${API}/auth/me`);
      setUser(response.data);
    } catch (error) {
      console.error('Auth check failed:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      const response = await axios.post(`${API}/auth/login`, { email, password });
      const { access_token, user: userData } = response.data;
      
      localStorage.setItem('token', access_token);
      setToken(access_token);
      setUser(userData);
      axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Login failed'
      };
    }
  };

  const register = async (userData) => {
    try {
      await axios.post(`${API}/auth/register`, userData);
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Registration failed'
      };
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    delete axios.defaults.headers.common['Authorization'];
  };

  return (
    <AuthContext.Provider value={{
      user,
      token,
      loading,
      login,
      register,
      logout
    }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Main App Component
function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

function AppContent() {
  const { user, loading } = useAuth();
  const [activeTab, setActiveTab] = useState('predict');

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return <AuthPage />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Navigation Header */}
      <nav className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  💼 SalaryPredict Pro
                </h1>
              </div>
              <div className="ml-8">
                <span className="text-sm text-gray-600">
                  Welcome, <span className="font-medium text-gray-800">{user.name}</span>
                  <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full capitalize">
                    {user.role.replace('_', ' ')}
                  </span>
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-8">
              <TabButton 
                active={activeTab === 'predict'} 
                onClick={() => setActiveTab('predict')}
                text="Salary Prediction"
                icon="🎯"
              />
              {(user.role === 'admin' || user.role === 'financial_analyst') && (
                <TabButton 
                  active={activeTab === 'compare'} 
                  onClick={() => setActiveTab('compare')}
                  text="Model Comparison"
                  icon="📊"
                />
              )}
              {user.role === 'admin' && (
                <TabButton 
                  active={activeTab === 'employees'} 
                  onClick={() => setActiveTab('employees')}
                  text="Employee Management"
                  icon="👥"
                />
              )}
              <TabButton 
                active={activeTab === 'tasks'} 
                onClick={() => setActiveTab('tasks')}
                text="My Tasks"
                icon="✅"
              />
              {user.role === 'admin' && (
                <TabButton 
                  active={activeTab === 'meetings'} 
                  onClick={() => setActiveTab('meetings')}
                  text="Meetings"
                  icon="📅"
                />
              )}
              <LogoutButton />
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {activeTab === 'predict' && <PredictionTab />}
        {activeTab === 'compare' && (user.role === 'admin' || user.role === 'financial_analyst') && <ComparisonTab />}
        {activeTab === 'employees' && user.role === 'admin' && <EmployeeManagementTab />}
        {activeTab === 'tasks' && <TasksTab />}
        {activeTab === 'meetings' && user.role === 'admin' && <MeetingsTab />}
      </main>
    </div>
  );
}

// Auth Page Component
const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    username: '',
    name: '',
    role: 'employee',
    skills: [],
    nomEntreprise: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [availableSkills, setAvailableSkills] = useState([]);
  const [customSkill, setCustomSkill] = useState('');
  const { login, register } = useAuth();

  useEffect(() => {
    if (!isLogin) {
      fetchSkills();
    }
  }, [isLogin]);

  const fetchSkills = async () => {
    try {
      const response = await axios.get(`${API}/skills/all`);
      const skillOptions = response.data.skills.map(skill => ({
        value: skill,
        label: skill
      }));
      setAvailableSkills(skillOptions);
    } catch (error) {
      console.error('Error fetching skills:', error);
    }
  };

  const addCustomSkill = async () => {
    if (!customSkill.trim()) return;
    
    try {
      await axios.post(`${API}/skills/add`, null, {
        params: { skill_name: customSkill.trim() }
      });
      
      // Add to current selection
      const newSkill = { value: customSkill.trim(), label: customSkill.trim() };
      setAvailableSkills(prev => [...prev, newSkill]);
      setFormData(prev => ({ ...prev, skills: [...prev.skills, customSkill.trim()] }));
      setCustomSkill('');
    } catch (error) {
      console.error('Error adding custom skill:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Validation
    if (!formData.email || !formData.password) {
      setError('Email and password are required');
      setLoading(false);
      return;
    }

    if (!isLogin && (!formData.username || formData.username.length < 3)) {
      setError('Username must be at least 3 characters long');
      setLoading(false);
      return;
    }

    if (!isLogin && (!formData.name || formData.name.length < 3)) {
      setError('Name must be at least 3 characters long');
      setLoading(false);
      return;
    }

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters long');
      setLoading(false);
      return;
    }

    try {
      let result;
      if (isLogin) {
        result = await login(formData.email, formData.password);
      } else {
        result = await register(formData);
        if (result.success) {
          setIsLogin(true);
          setError('');
          setFormData({
            email: formData.email,
            password: '',
            username: '',
            name: '',
            role: 'employee'
          });
          alert('Registration successful! Please login.');
          setLoading(false);
          return;
        }
      }

      if (!result.success) {
        setError(result.error);
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (error) setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-2">
            💼 SalaryPredict Pro
          </h1>
          <p className="text-gray-600">
            {isLogin ? 'Sign in to your account' : 'Create your account'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700">
              {error}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Email Address *
            </label>
            <input
              type="email"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              required
            />
          </div>

          {!isLogin && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Username *
                </label>
                <input
                  type="text"
                  value={formData.username}
                  onChange={(e) => handleInputChange('username', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  minLength={3}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => handleInputChange('name', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  minLength={3}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Role *
                </label>
                <select
                  value={formData.role}
                  onChange={(e) => handleInputChange('role', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                >
                  <option value="employee">Employee</option>
                  <option value="admin">Admin</option>
                  <option value="financial_analyst">Financial Analyst</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Skills
                </label>
                <Select
                  isMulti
                  value={availableSkills.filter(skill => formData.skills.includes(skill.value))}
                  onChange={(selectedOptions) => {
                    const selectedSkills = selectedOptions ? selectedOptions.map(option => option.value) : [];
                    handleInputChange('skills', selectedSkills);
                  }}
                  options={availableSkills}
                  className="basic-multi-select"
                  classNamePrefix="select"
                  placeholder="Select your skills..."
                />
                <div className="mt-2 flex gap-2">
                  <input
                    type="text"
                    value={customSkill}
                    onChange={(e) => setCustomSkill(e.target.value)}
                    placeholder="Add custom skill"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <button
                    type="button"
                    onClick={addCustomSkill}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Add
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Company Name
                </label>
                <input
                  type="text"
                  value={formData.nomEntreprise}
                  onChange={(e) => handleInputChange('nomEntreprise', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter your company name"
                />
              </div>
            </>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Password *
            </label>
            <input
              type="password"
              value={formData.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              minLength={6}
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-6 rounded-lg font-medium hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 disabled:opacity-50"
          >
            {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Sign Up')}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="text-blue-600 hover:text-blue-700"
          >
            {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
          </button>
        </div>
      </div>
    </div>
  );
};

// UI Components
const TabButton = ({ active, onClick, text, icon }) => (
  <button
    onClick={onClick}
    className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
      active 
        ? 'bg-blue-100 text-blue-700 border border-blue-200' 
        : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
    }`}
  >
    <span>{icon}</span>
    <span className="font-medium">{text}</span>
  </button>
);

const LogoutButton = () => {
  const { logout } = useAuth();
  
  return (
    <button
      onClick={logout}
      className="flex items-center space-x-2 px-4 py-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-all duration-200"
    >
      <span>🚪</span>
      <span className="font-medium">Logout</span>
    </button>
  );
};

// Tab Components (keeping existing prediction and comparison tabs)
const PredictionTab = () => {
  const [formData, setFormData] = useState({
    work_year: 2024,
    experience_level: '',
    employment_type: '',
    job_title: '',
    employee_residence: '',
    remote_ratio: 0,
    company_location: '',
    company_size: ''
  });
  const [options, setOptions] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchOptions();
  }, []);

  const fetchOptions = async () => {
    try {
      const response = await axios.get(`${API}/available-options`);
      setOptions(response.data);
    } catch (error) {
      console.error('Error fetching options:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await axios.post(`${API}/predict-salary`, formData);
      setPrediction(response.data);
    } catch (error) {
      console.error('Error predicting salary:', error);
      if (error.response?.status === 401) {
        alert('Your session has expired. Please login again.');
      } else {
        alert('Error predicting salary. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Prediction Form */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <span className="mr-3">🎯</span>
          Salary Prediction
        </h2>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Work Year */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Work Year
              </label>
              <input
                type="number"
                min="2020"
                max="2025"
                value={formData.work_year}
                onChange={(e) => handleInputChange('work_year', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>

            {/* Experience Level */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Experience Level
              </label>
              <select
                value={formData.experience_level}
                onChange={(e) => handleInputChange('experience_level', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="">Select Experience Level</option>
                {options.experience_levels?.map(level => (
                  <option key={level} value={level}>
                    {options.experience_level_descriptions?.[level] || level}
                  </option>
                ))}
              </select>
            </div>

            {/* Employment Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Employment Type
              </label>
              <select
                value={formData.employment_type}
                onChange={(e) => handleInputChange('employment_type', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="">Select Employment Type</option>
                {options.employment_types?.map(type => (
                  <option key={type} value={type}>
                    {options.employment_type_descriptions?.[type] || type}
                  </option>
                ))}
              </select>
            </div>

            {/* Job Title */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Job Title
              </label>
              <select
                value={formData.job_title}
                onChange={(e) => handleInputChange('job_title', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="">Select Job Title</option>
                {options.job_titles?.map(title => (
                  <option key={title} value={title}>{title}</option>
                ))}
              </select>
            </div>

            {/* Employee Residence */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Employee Residence
              </label>
              <select
                value={formData.employee_residence}
                onChange={(e) => handleInputChange('employee_residence', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="">Select Country</option>
                {options.countries?.map(country => (
                  <option key={country} value={country}>{country}</option>
                ))}
              </select>
            </div>

            {/* Company Location */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Company Location
              </label>
              <select
                value={formData.company_location}
                onChange={(e) => handleInputChange('company_location', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="">Select Country</option>
                {options.countries?.map(country => (
                  <option key={country} value={country}>{country}</option>
                ))}
              </select>
            </div>

            {/* Remote Ratio */}
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Remote Work Ratio: {formData.remote_ratio}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                step="25"
                value={formData.remote_ratio}
                onChange={(e) => handleInputChange('remote_ratio', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-gray-500 mt-1">
                <span>0% (On-site)</span>
                <span>50% (Hybrid)</span>
                <span>100% (Remote)</span>
              </div>
            </div>

            {/* Company Size */}
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Company Size
              </label>
              <select
                value={formData.company_size}
                onChange={(e) => handleInputChange('company_size', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              >
                <option value="">Select Company Size</option>
                {options.company_sizes?.map(size => (
                  <option key={size} value={size}>
                    {options.company_size_descriptions?.[size] || size}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-6 rounded-lg font-medium hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 disabled:opacity-50"
          >
            {loading ? 'Predicting...' : 'Predict Salary 🚀'}
          </button>
        </form>
      </div>

      {/* Prediction Results */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <span className="mr-3">💰</span>
          Prediction Results
        </h2>
        
        {prediction ? (
          <div className="space-y-6">
            <div className="text-center bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6">
              <div className="text-4xl font-bold text-green-600 mb-2">
                ${prediction.predicted_salary_usd.toLocaleString()}
              </div>
              <div className="text-lg text-gray-600 mb-4">Predicted Annual Salary (USD)</div>
              <div className="flex justify-center space-x-4 text-sm">
                <div className="bg-white px-3 py-1 rounded-full">
                  <span className="text-gray-600">Model: </span>
                  <span className="font-medium">{prediction.model_name}</span>
                </div>
                <div className="bg-white px-3 py-1 rounded-full">
                  <span className="text-gray-600">Confidence: </span>
                  <span className="font-medium">{(prediction.confidence_score * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-800 mb-3">Input Summary</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><span className="text-gray-600">Experience:</span> <span className="font-medium">{options.experience_level_descriptions?.[prediction.input_data.experience_level]}</span></div>
                <div><span className="text-gray-600">Employment:</span> <span className="font-medium">{options.employment_type_descriptions?.[prediction.input_data.employment_type]}</span></div>
                <div><span className="text-gray-600">Job Title:</span> <span className="font-medium">{prediction.input_data.job_title}</span></div>
                <div><span className="text-gray-600">Company Size:</span> <span className="font-medium">{options.company_size_descriptions?.[prediction.input_data.company_size]}</span></div>
                <div><span className="text-gray-600">Location:</span> <span className="font-medium">{prediction.input_data.company_location}</span></div>
                <div><span className="text-gray-600">Remote:</span> <span className="font-medium">{prediction.input_data.remote_ratio}%</span></div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <div className="text-6xl mb-4">📊</div>
            <p>Fill out the form to get salary prediction</p>
          </div>
        )}
      </div>
    </div>
  );
};

// Model Comparison Tab
const ComparisonTab = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API}/models-comparison`);
      setModels(response.data);
    } catch (error) {
      console.error('Error fetching models:', error);
      if (error.response?.status === 403) {
        alert('Access denied. This feature is only available for admins and financial analysts.');
      }
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score, type = 'r2') => {
    if (type === 'r2') {
      if (score >= 0.8) return 'text-green-600';
      if (score >= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    }
    return 'text-gray-800';
  };

  const formatMetric = (value, type) => {
    if (type === 'time') return `${value.toFixed(3)}s`;
    if (type === 'currency') return `$${value.toFixed(0)}`;
    return value.toFixed(4);
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-xl text-gray-600">Loading model comparisons...</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        <span className="mr-3">📊</span>
        Model Performance Comparison
      </h2>

      <div className="overflow-x-auto">
        <table className="w-full table-auto">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-4 px-4 font-semibold text-gray-800">Model</th>
              <th className="text-center py-4 px-4 font-semibold text-gray-800">R² Score</th>
              <th className="text-center py-4 px-4 font-semibold text-gray-800">MAE</th>
              <th className="text-center py-4 px-4 font-semibold text-gray-800">RMSE</th>
              <th className="text-center py-4 px-4 font-semibold text-gray-800">MSE</th>
              <th className="text-center py-4 px-4 font-semibold text-gray-800">Training Time</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model, index) => (
              <tr key={model.model_name} className={`${index === 0 ? 'bg-green-50 border-green-200' : ''} border-b border-gray-100 hover:bg-gray-50`}>
                <td className="py-4 px-4">
                  <div className="flex items-center">
                    {index === 0 && <span className="text-green-500 mr-2">👑</span>}
                    <span className="font-medium">{model.model_name}</span>
                    {index === 0 && <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">Best</span>}
                  </div>
                </td>
                <td className={`text-center py-4 px-4 font-medium ${getScoreColor(model.r2_score)}`}>
                  {formatMetric(model.r2_score, 'score')}
                </td>
                <td className="text-center py-4 px-4">{formatMetric(model.mae, 'currency')}</td>
                <td className="text-center py-4 px-4">{formatMetric(model.rmse, 'currency')}</td>
                <td className="text-center py-4 px-4">{formatMetric(model.mse, 'large')}</td>
                <td className="text-center py-4 px-4 text-gray-600">{formatMetric(model.training_time, 'time')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-medium text-blue-800 mb-2">📚 Metrics Explanation</h4>
        <div className="text-sm text-blue-700 space-y-1">
          <p><strong>R² Score:</strong> Coefficient of determination (higher is better, max 1.0)</p>
          <p><strong>MAE:</strong> Mean Absolute Error (lower is better)</p>
          <p><strong>RMSE:</strong> Root Mean Squared Error (lower is better)</p>
          <p><strong>MSE:</strong> Mean Squared Error (lower is better)</p>
        </div>
      </div>
    </div>
  );
};

// Employee Management Tab (Admin only)
const EmployeeManagementTab = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        <span className="mr-3">👥</span>
        Employee Management
      </h2>
      <div className="text-center py-12 text-gray-500">
        <div className="text-6xl mb-4">🚧</div>
        <p className="text-xl mb-2">Employee Management Coming Soon</p>
        <p>CRUD operations, team management, and employee analytics will be available here</p>
      </div>
    </div>
  );
};

// Tasks Tab
const TasksTab = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        <span className="mr-3">✅</span>
        My Tasks
      </h2>
      <div className="text-center py-12 text-gray-500">
        <div className="text-6xl mb-4">📋</div>
        <p className="text-xl mb-2">Task Management Coming Soon</p>
        <p>View and manage your assigned tasks here</p>
      </div>
    </div>
  );
};

// Meetings Tab (Admin only)
const MeetingsTab = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        <span className="mr-3">📅</span>
        Meeting Management
      </h2>
      <div className="text-center py-12 text-gray-500">
        <div className="text-6xl mb-4">🗓️</div>
        <p className="text-xl mb-2">Meeting Management Coming Soon</p>
        <p>Create, schedule, and manage meetings here</p>
      </div>
    </div>
  );
};

export default App;