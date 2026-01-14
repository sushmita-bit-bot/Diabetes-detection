// DOM Elements
const predictionForm = document.getElementById('prediction-form');
const loadingElement = document.getElementById('loading');
const resultElement = document.getElementById('result');
const resultIcon = document.getElementById('result-icon');
const resultTitle = document.getElementById('result-title');
const resultProbability = document.getElementById('result-probability');
const probabilityFill = resultProbability.querySelector('.probability-fill');
const probabilityText = resultProbability.querySelector('.probability-text');
const modelUsed = document.getElementById('model-used');

// API URL
const API_URL = 'http://localhost:5000';

// Event Listeners
predictionForm.addEventListener('submit', handlePrediction);

// Functions
async function handlePrediction(e) {
    e.preventDefault();
    
    // Show loading, hide result
    loadingElement.style.display = 'flex';
    resultElement.style.display = 'none';
    
    // Get form data
    const formData = new FormData(predictionForm);
    const data = {
        pregnancies: formData.get('pregnancies'),
        glucose: formData.get('glucose'),
        bloodPressure: formData.get('bloodPressure'),
        skinThickness: formData.get('skinThickness'),
        insulin: formData.get('insulin'),
        bmi: formData.get('bmi'),
        diabetesPedigree: formData.get('diabetesPedigree'),
        age: formData.get('age'),
        model: formData.get('model')
    };
    
    try {
        // Make API call
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Failed to get prediction');
        }
        
        const result = await response.json();
        displayResult(result);
    } catch (error) {
        console.error('Error:', error);
        displayError(error.message);
    } finally {
        // Hide loading
        loadingElement.style.display = 'none';
        resultElement.style.display = 'block';
    }
}

function displayResult(result) {
    const { prediction, probability, model } = result;
    const probabilityPercentage = Math.round(probability * 100);
    
    // Update UI elements
    if (prediction === 1) {
        resultIcon.innerHTML = '⚠️';
        resultTitle.textContent = 'Diabetes Risk Detected';
        resultTitle.style.color = '#e74c3c';
    } else {
        resultIcon.innerHTML = '✅';
        resultTitle.textContent = 'No Diabetes Risk Detected';
        resultTitle.style.color = '#2ecc71';
    }
    
    // Update probability bar
    probabilityFill.style.width = `${probabilityPercentage}%`;
    probabilityText.textContent = `${probabilityPercentage}%`;
    
    // Change color based on probability
    if (probabilityPercentage > 70) {
        probabilityFill.style.backgroundColor = '#e74c3c';
    } else if (probabilityPercentage > 30) {
        probabilityFill.style.backgroundColor = '#f39c12';
    } else {
        probabilityFill.style.backgroundColor = '#2ecc71';
    }
    
    // Update model info
    modelUsed.textContent = formatModelName(model);
    
    // Show result details
    const resultDetails = document.querySelector('.result-details p');
    resultDetails.textContent = prediction === 1 
        ? `There is a ${probabilityPercentage}% probability of diabetes based on the provided health metrics.`
        : `There is a ${probabilityPercentage}% probability of diabetes based on the provided health metrics.`;
}

function displayError(message) {
    resultIcon.innerHTML = '❌';
    resultTitle.textContent = 'Error';
    resultTitle.style.color = '#e74c3c';
    
    const resultDetails = document.querySelector('.result-details p');
    resultDetails.textContent = `An error occurred: ${message}. Please try again.`;
    
    probabilityFill.style.width = '0%';
    probabilityText.textContent = 'N/A';
    modelUsed.textContent = 'None';
}

function formatModelName(model) {
    switch(model) {
        case 'logistic_regression':
            return 'Logistic Regression';
        case 'random_forest':
            return 'Random Forest';
        case 'svm':
            return 'Support Vector Machine';
        case 'ensemble':
            return 'Ensemble (Combined Models)';
        default:
            return model;
    }
}

// Health check to verify API is running
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('API is running');
        } else {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.error('API is not available:', error);
    }
}

// Check API health on page load
window.addEventListener('load', checkApiHealth);