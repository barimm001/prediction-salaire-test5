import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Main App Component
function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [loading, setLoading] = useState(false);

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
            </div>
            <div className="flex items-center space-x-8">
              <TabButton 
                active={activeTab === 'predict'} 
                onClick={() => setActiveTab('predict')}
                text="Salary Prediction"
                icon="🎯"
              />
              <TabButton 
                active={activeTab === 'compare'} 
                onClick={() => setActiveTab('compare')}
                text="Model Comparison"
                icon="📊"
              />
              <TabButton 
                active={activeTab === 'dashboard'} 
                onClick={() => setActiveTab('dashboard')}
                text="Dashboard"
                icon="📈"
              />
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {activeTab === 'predict' && <PredictionTab />}
        {activeTab === 'compare' && <ComparisonTab />}
        {activeTab === 'dashboard' && <DashboardTab />}
      </main>
    </div>
  );
}

// Tab Button Component
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

// Salary Prediction Tab
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
      alert('Error predicting salary. Please try again.');
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

// Dashboard Tab (placeholder)
const DashboardTab = () => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
        <span className="mr-3">📈</span>
        Analytics Dashboard
      </h2>
      <div className="text-center py-12 text-gray-500">
        <div className="text-6xl mb-4">🚧</div>
        <p className="text-xl mb-2">Dashboard Coming Soon</p>
        <p>Interactive charts and analytics will be available here</p>
      </div>
    </div>
  );
};

export default App;