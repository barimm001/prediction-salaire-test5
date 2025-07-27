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
              {user.role === 'financial_analyst' && (
                <TabButton 
                  active={activeTab === 'analytics'} 
                  onClick={() => setActiveTab('analytics')}
                  text="Advanced Analytics"
                  icon="📈"
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
        {activeTab === 'analytics' && user.role === 'financial_analyst' && <AnalyticsTab />}
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
            role: 'employee',
            skills: [],
            nomEntreprise: ''
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
    company_size: '',
    skills: [],
    nomEntreprise: ''
  });
  const [options, setOptions] = useState({});
  const [availableSkills, setAvailableSkills] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchOptions();
    fetchSkills();
  }, []);

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
            <div>
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

            {/* Skills */}
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
            </div>

            {/* Company Name */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Company Name
              </label>
              <input
                type="text"
                value={formData.nomEntreprise}
                onChange={(e) => handleInputChange('nomEntreprise', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter company name"
                required
              />
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

// Advanced Analytics Tab (Financial Analyst only)
const AnalyticsTab = () => {
  const [analyticsData, setAnalyticsData] = useState({
    salaryTrends: [],
    companySummaries: [],
    correlationMatrix: [],
    topRankings: {},
    annualSummaries: [],
    salaryDistribution: {}
  });
  const [loading, setLoading] = useState(true);
  const [activeChart, setActiveChart] = useState('trends');

  useEffect(() => {
    fetchAllAnalytics();
  }, []);

  const fetchAllAnalytics = async () => {
    try {
      setLoading(true);
      
      const [trendsRes, summariesRes, correlationRes, rankingsRes, annualRes, distributionRes] = 
        await Promise.all([
          axios.get(`${API}/analytics/salary-trends`),
          axios.get(`${API}/analytics/company-summaries`),
          axios.get(`${API}/analytics/correlation-heatmap`),
          axios.get(`${API}/analytics/top-rankings`),
          axios.get(`${API}/analytics/annual-summary`),
          axios.get(`${API}/analytics/salary-distribution`)
        ]);

      setAnalyticsData({
        salaryTrends: trendsRes.data.trends || [],
        companySummaries: summariesRes.data.summaries || [],
        correlationMatrix: correlationRes.data.correlation_matrix || [],
        topRankings: rankingsRes.data,
        annualSummaries: annualRes.data.annual_summaries || [],
        salaryDistribution: distributionRes.data
      });
    } catch (error) {
      console.error('Error fetching analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FFC658', '#FF7C80'];

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-xl text-gray-600">Loading advanced analytics...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Navigation Pills */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <span className="mr-3">📈</span>
          Advanced Financial Analytics
        </h2>
        <div className="flex flex-wrap gap-2">
          {[
            { key: 'trends', label: 'Salary Trends', icon: '📈' },
            { key: 'companies', label: 'Company Analysis', icon: '🏢' },
            { key: 'correlations', label: 'Correlations', icon: '🔗' },
            { key: 'rankings', label: 'Top Rankings', icon: '🏆' },
            { key: 'annual', label: 'Annual Summary', icon: '📅' },
            { key: 'distribution', label: 'Salary Distribution', icon: '📊' }
          ].map(chart => (
            <button
              key={chart.key}
              onClick={() => setActiveChart(chart.key)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                activeChart === chart.key 
                  ? 'bg-blue-100 text-blue-700 border border-blue-200' 
                  : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
              }`}
            >
              <span>{chart.icon}</span>
              <span className="font-medium">{chart.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Salary Trends Chart */}
      {activeChart === 'trends' && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">📈 Salary Trends by Job & Company</h3>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={analyticsData.salaryTrends.slice(0, 20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="job_title" 
                  angle={-45} 
                  textAnchor="end" 
                  height={100}
                  fontSize={12}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value) => [`$${value.toLocaleString()}`, 'Average Salary']}
                  labelFormatter={(label) => `Job: ${label}`}
                />
                <Bar dataKey="avg_salary" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Company Analysis */}
      {activeChart === 'companies' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">🏢 Total Salary Cost by Company</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analyticsData.companySummaries.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="company" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Total Cost']} />
                  <Bar dataKey="total_salary_cost" fill="#00C49F" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">💰 Average Salary by Company</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analyticsData.companySummaries.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="company" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Average Salary']} />
                  <Bar dataKey="avg_salary" fill="#FFBB28" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Top Rankings */}
      {activeChart === 'rankings' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Top Jobs */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">🏆 Top Paying Jobs</h3>
            <div className="space-y-3">
              {analyticsData.topRankings.top_jobs?.slice(0, 8).map((job, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <div>
                    <div className="font-medium text-sm">{job.job_title}</div>
                    <div className="text-xs text-gray-500">{job.count} positions</div>
                  </div>
                  <div className="text-green-600 font-bold">${job.avg_salary.toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Companies */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">🏢 Top Paying Companies</h3>
            <div className="space-y-3">
              {analyticsData.topRankings.top_companies?.slice(0, 8).map((company, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <div>
                    <div className="font-medium text-sm">{company.company}</div>
                    <div className="text-xs text-gray-500">{company.count} employees</div>
                  </div>
                  <div className="text-blue-600 font-bold">${company.avg_salary.toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Skills */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">🔥 Most Demanded Skills</h3>
            <div className="space-y-3">
              {analyticsData.topRankings.top_skills?.slice(0, 8).map((skill, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <div>
                    <div className="font-medium text-sm">{skill.skill}</div>
                    <div className="text-xs text-gray-500">{skill.count} mentions</div>
                  </div>
                  <div className="text-purple-600 font-bold">${skill.avg_salary.toLocaleString()}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Annual Summary */}
      {activeChart === 'annual' && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4">📅 Annual Summary & Growth</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-80">
              <h4 className="text-lg font-medium mb-2">Recruitment Trends</h4>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analyticsData.annualSummaries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="total_recruitments" stroke="#8884d8" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="h-80">
              <h4 className="text-lg font-medium mb-2">Salary Evolution</h4>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analyticsData.annualSummaries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Average Salary']} />
                  <Line type="monotone" dataKey="avg_salary" stroke="#00C49F" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Salary Distribution */}
      {activeChart === 'distribution' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">📊 Salary Distribution by Company Size</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analyticsData.salaryDistribution.distribution_by_company_size}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`]} />
                  <Bar dataKey="mean" fill="#FF8042" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">💼 Salary Distribution by Experience</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analyticsData.salaryDistribution.distribution_by_experience}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toLocaleString()}`]} />
                  <Bar dataKey="mean" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6 lg:col-span-2">
            <h3 className="text-xl font-bold text-gray-800 mb-4">📈 Overall Salary Histogram</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analyticsData.salaryDistribution.salary_histogram}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range_label" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip formatter={(value) => [value, 'Number of Employees']} />
                  <Bar dataKey="count" fill="#00C49F" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <div className="text-3xl font-bold text-blue-600">
            {analyticsData.companySummaries.reduce((sum, c) => sum + c.total_employees, 0)}
          </div>
          <div className="text-gray-600">Total Employees</div>
        </div>
        
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <div className="text-3xl font-bold text-green-600">
            ${analyticsData.companySummaries.reduce((sum, c) => sum + c.total_salary_cost, 0).toLocaleString()}
          </div>
          <div className="text-gray-600">Total Salary Cost</div>
        </div>
        
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <div className="text-3xl font-bold text-purple-600">
            {analyticsData.companySummaries.length}
          </div>
          <div className="text-gray-600">Companies</div>
        </div>
        
        <div className="bg-white rounded-xl shadow-lg p-6 text-center">
          <div className="text-3xl font-bold text-red-600">
            ${Math.round(analyticsData.companySummaries.reduce((sum, c) => sum + c.avg_salary, 0) / Math.max(analyticsData.companySummaries.length, 1)).toLocaleString()}
          </div>
          <div className="text-gray-600">Average Salary</div>
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