# 🌾 Smart Agriculture Assistant - Full Stack Application

AI-Powered Soil Analysis & Crop Recommendation System with Dynamic Crop Suggestions based on Soil Visual Characteristics.

## 🎯 Features

- **Soil Type Detection**: CNN-based classification (Alluvial, Black, Clay, Red soil)
- **Soil Health Index**: Automated calculation (0-100 scale)
- **Dynamic Crop Recommendations**: AI-powered suggestions based on soil color intensity and characteristics
- **Weather Integration**: OpenWeather API for location-based climate data
- **Fertilizer Calculator**: NPK deficiency analysis and quantity recommendations
- **AI Chat Assistant**: Mistral AI chatbot for agricultural queries
- **PDF Report Generation**: Comprehensive soil analysis reports

## 📋 Prerequisites

- Python 3.8+
- Node.js (for frontend development server - optional)
- Valid API Keys:
  - Mistral AI API Key
  - OpenWeather API Key

## 🚀 Installation & Setup

### Step 1: Clone/Download the Project

Create a project directory and add these files:
```
smart-agriculture/
├── app.py                 # Flask backend
├── index.html            # Frontend interface
├── requirements.txt      # Python dependencies
├── models/               # (Create this folder)
│   └── soil_texture_cnn.keras  # Your trained model
└── README.md
```

### Step 2: Install Python Dependencies

Create `requirements.txt`:

```txt
flask==3.0.0
flask-cors==4.0.0
tensorflow==2.15.0
opencv-python==4.8.1.78
numpy==1.24.3
scikit-image==0.22.0
reportlab==4.0.7
mistralai==0.1.0
Pillow==10.1.0
requests==2.31.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Open `app.py` and update these lines:

```python
# Line 23-24
OPENWEATHER_KEY = "YOUR_OPENWEATHER_API_KEY_HERE"
MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY_HERE"
```

**Get API Keys:**
- **Mistral AI**: https://console.mistral.ai/
- **OpenWeather**: https://openweathermap.org/api

### Step 4: Train or Load CNN Model

**Option A: Use Pre-trained Model**
- Place your `soil_texture_cnn.keras` file in the `models/` folder
- Update line 65 in `app.py` to load from correct path

**Option B: Train New Model**
- Prepare dataset with folders: `Alluvial Soil/`, `Black Soil/`, `Clay Soil/`, `Red Soil/`
- Add training code or use Google Colab notebook

### Step 5: Run the Backend

```bash
python app.py
```

Expected output:
```
* Running on http://127.0.0.1:5000
* Debug mode: on
```

### Step 6: Open the Frontend

**Option A: Direct File Access**
- Simply open `index.html` in your web browser
- Update API_BASE in JavaScript if needed (line 603)

**Option B: Local Server (Recommended)**
```bash
# Using Python
python -m http.server 8000

# Then open: http://localhost:8000
```

**Option C: VS Code Live Server**
- Install "Live Server" extension
- Right-click `index.html` → "Open with Live Server"

## 🎮 Usage Guide

### 1. Upload Soil Image
- Navigate to "Upload Soil Image" tab
- Select a soil image from your device
- (Optional) Enter location for weather data
- (Optional) Enter soil moisture percentage
- Click "Analyze Soil"

### 2. View Soil Analysis
- Switch to "Soil Analysis" tab
- View detected soil type and health index
- Check image features and weather data

### 3. Get Crop Recommendations
- Go to "Crop Recommendations" tab
- Click "Get Crop Recommendations"
- View AI-generated crop suggestions based on your soil

### 4. Chat with AI Assistant
- Navigate to "AI Chat Assistant" tab
- Ask questions about:
  - Suitable crops for your soil type
  - Seasonal farming practices
  - Drought-resistant options
  - High-value crops
  - Irrigation requirements

### 5. Calculate Fertilizer Needs
- Go to "Fertilizer Calculator" tab
- Enter your land size and unit
- Input current NPK values (or use defaults)
- Select fertilizer type
- Click "Calculate Fertilizer Quantity"
- View detailed recommendations and cost estimates

## 🔧 Configuration

### Backend Configuration (app.py)

```python
# Port configuration (line 695)
app.run(debug=True, port=5000)

# CORS settings (line 21)
CORS(app)  # Allow all origins

# For specific origins:
CORS(app, origins=["http://localhost:8000"])
```

### Frontend Configuration (index.html)

```javascript
// API endpoint (line 603)
const API_BASE = 'http://localhost:5000/api';

// For production:
const API_BASE = 'https://your-domain.com/api';
```

## 📊 Fertilizer Database

The system includes 10 common fertilizers:

| Fertilizer | N% | P% | K% | Price (₹/kg) |
|------------|----|----|----|--------------| 
| Urea | 46 | 0 | 0 | 6 |
| DAP | 18 | 46 | 0 | 27 |
| MOP | 0 | 0 | 60 | 17 |
| NPK 10-26-26 | 10 | 26 | 26 | 24 |
| NPK 19-19-19 | 19 | 19 | 19 | 20 |

*Prices can be updated in `app.py` lines 28-38*

## 🧪 API Endpoints

### POST `/api/analyze-soil`
Upload and analyze soil image
- **Body**: FormData with `image`, `location` (optional), `moisture` (optional)
- **Returns**: Session ID, soil type, features, health score

### POST `/api/crop-recommendations`
Get AI-powered crop suggestions
- **Body**: `{ session_id }`
- **Returns**: Crop recommendations text

### POST `/api/chat`
Chat with AI assistant
- **Body**: `{ session_id, message, history }`
- **Returns**: AI response

### POST `/api/calculate-fertilizer`
Calculate fertilizer quantities
- **Body**: `{ session_id, land_size, land_unit, n, p, k, fertilizer }`
- **Returns**: Calculation results

### GET `/api/fertilizers`
Get available fertilizer database
- **Returns**: JSON of all fertilizers

## 🐛 Troubleshooting

### CORS Errors
```bash
# Install flask-cors
pip install flask-cors

# Add to app.py:
from flask_cors import CORS
CORS(app)
```

### API Connection Failed
- Ensure backend is running on port 5000
- Check firewall settings
- Update `API_BASE` in frontend

### Model Not Found
- Verify `soil_texture_cnn.keras` exists
- Check file path in code
- Or train a new model using your dataset

### Mistral AI Errors
- Verify API key is correct
- Check internet connection
- Ensure account has credits

### OpenWeather API Errors
- Confirm API key is active
- Check location name format
- Free tier has rate limits (60 calls/min)

## 🔐 Security Notes

### ⚠️ IMPORTANT: API Key Security

**For Development:**
- Current setup has keys in code (NOT RECOMMENDED for production)

**For Production:**
1. **Use Environment Variables:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
OPENWEATHER_KEY = os.getenv('OPENWEATHER_KEY')
```

2. **Create `.env` file:**
```
MISTRAL_API_KEY=your_key_here
OPENWEATHER_KEY=your_key_here
```

3. **Add to `.gitignore`:**
```
.env
*.keras
__pycache__/
```

## 📦 Deployment

### Backend (Flask)

**Option 1: Heroku**
```bash
# Create Procfile
web: gunicorn app:app

# Deploy
git init
heroku create your-app-name
git push heroku main
```

**Option 2: PythonAnywhere**
- Upload files via web interface
- Configure WSGI file
- Set environment variables

**Option 3: Docker**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Frontend

**Option 1: Netlify/Vercel**
- Drag and drop `index.html`
- Update API_BASE to backend URL

**Option 2: GitHub Pages**
```bash
git add .
git commit -m "Deploy"
git push origin main
```

## 🎓 Model Training (Optional)

To train your own CNN model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prepare dataset structure:
# dataset/
#   ├── Alluvial Soil/
#   ├── Black Soil/
#   ├── Clay Soil/
#   └── Red Soil/

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)
model.save('soil_texture_cnn.keras')
```

## 📚 Resources

- **Mistral AI Docs**: https://docs.mistral.ai/
- **OpenWeather API**: https://openweathermap.org/api
- **TensorFlow**: https://www.tensorflow.org/
- **Flask**: https://flask.palletsprojects.com/

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 💬 Support

For issues or questions:
- Check troubleshooting section
- Review API documentation
- Contact: your-email@example.com

## 🎉 Acknowledgments

- Mistral AI for LLM capabilities
- OpenWeather for weather data
- TensorFlow team for ML framework
- Agricultural research community

---

**Built with ❤️ for Indian Farmers** 🇮🇳

*Version 1.0.0 - Last Updated: 2024*