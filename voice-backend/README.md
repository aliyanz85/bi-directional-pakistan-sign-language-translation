# D-VOICE Animation Service Backend

FastAPI-based backend service for managing PSL (Pakistan Sign Language) animation metadata and serving animation-related endpoints to the frontend.

## 🚀 Features

- **Animation Metadata Management**: Centralized registry of 32 PSL sign animations
- **RESTful API**: Clean endpoints for fetching animation data
- **Animation Resolution**: Smart phrase-to-animation matching (simple word matching for now, expandable for ML)
- **CORS Enabled**: Ready for local frontend development
- **Auto-documented**: FastAPI auto-generates interactive API docs

## 📁 Project Structure

```
voice-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── animations_config.json     # Animation metadata registry
│   ├── api/
│   │   ├── __init__.py
│   │   └── animations.py          # Animation endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py              # App configuration & settings
│   └── models/
│       └── __init__.py
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Setup Instructions

### 1. Create a Virtual Environment

```bash
# Navigate to voice-backend folder
cd voice-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Development Server

```bash
# Option 1: Using uvicorn directly
uvicorn app.main:app --reload --port 8000

# Option 2: Using Python
python -m app.main
```

The server will start at: **http://localhost:8000**

### 4. Verify Installation

Visit these URLs in your browser:
- **API Root**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🌍 Language Support

The backend now supports **Urdu language input**! When a user types in Urdu, the backend automatically translates it to English to match available animations.

**How it works:**
- User types: "خوش" (khush - happy/excited)
- Backend translates to: "excited"
- Returns animation for "excited"

Over 40 Urdu words are mapped to the 32 available animations. See [`app/urdu_mappings.py`](app/urdu_mappings.py) for the complete mapping.

## 📡 API Endpoints

### 1. Health Check
```
GET /health
```
Returns server status and version info.

**Response:**
```json
{
  "status": "ok",
  "service": "D-VOICE Animation Service",
  "version": "1.0.0"
}
```

### 2. List All Animations
```
GET /api/animations
```
Returns all available animations with metadata.

**Response:**
```json
[
  {
    "id": "alert",
    "name": "Alert",
    "description": "Sign for alert or warning",
    "file_path": "/animations/alert.glb",
    "category": "communication",
    "tags": ["warning", "attention", "important"]
  },
  ...
]
```

### 3. Get Animation by ID
```
GET /api/animations/{animation_id}
```
Returns a single animation's metadata.

**Example:**
```
GET /api/animations/hello
```

**Response:**
```json
{
  "id": "hello",
  "name": "Hello",
  "description": "Sign for hello or greeting",
  "file_path": "/animations/hello.glb",
  "category": "greetings",
  "tags": ["greeting", "welcome"]
}
```

### 4. Resolve Animation for Phrase
```
POST /api/resolve-animation
```
Finds the best animation for a given text phrase. Supports both English and Urdu.

**Request Body (English):**
```json
{
  "phrase": "I am excited",
  "language": "psl"
}
```

**Request Body (Urdu):**
```json
{
  "phrase": "میں خوش ہوں",
  "language": "ur"
}
```

**Response:**
```json
{
  "animation": {
    "id": "excited",
    "name": "Excited",
    "description": "Sign for excited or enthusiastic",
    "file_path": "/animations/excited.glb",
    "category": "emotions",
    "tags": ["emotion", "happy", "energetic"]
  },
  "matched_word": "excited",
  "confidence": 1.0
}
```

### 5. Get Urdu Words
```
GET /api/urdu-words
```
Returns Urdu display names for all available animations.

**Response:**
```json
{
  "excited": "خوش / پرجوش",
  "good": "اچھا",
  "smart": "ذہین / عقلمند",
  ...
}
```

## 🔧 Configuration

Configuration is managed in [`app/core/config.py`](app/core/config.py).

### CORS Settings
By default, these origins are allowed:
- `http://localhost:5173` (Vite dev server)
- `http://localhost:3000`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:3000`

To add more origins, edit the `CORS_ORIGINS` list in `config.py`.

### Environment Variables
Create a `.env` file in the `voice-backend` folder to override defaults:

```env
HOST=0.0.0.0
PORT=8000
```

## 📝 Animation Metadata

Animation metadata is stored in [`app/animations_config.json`](app/animations_config.json).

Each animation has:
- `id`: Unique identifier (used in API calls)
- `name`: Display name
- `description`: What the sign represents
- `file_path`: Relative path to GLB file (frontend will load this)
- `category`: Grouping (e.g., emotions, communication, adjectives)
- `tags`: Keywords for fuzzy matching

### Available Animations (32 total)
alert, book, careful, cheap, crazy, dangerous, decent, dumb, excited, extreme, fantastic, far, fearful, foreign, funny, good, healthy, heavy, important, intelligent, interesting, late, less, new, no, noisy, peaceful, quick, ready, secure, smart, yes

## 🧪 Testing the API

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Get all animations
curl http://localhost:8000/api/animations

# Get specific animation
curl http://localhost:8000/api/animations/excited

# Resolve animation for phrase
curl -X POST http://localhost:8000/api/resolve-animation \
  -H "Content-Type: application/json" \
  -d '{"phrase": "I am excited", "language": "psl"}'
```

### Using the Interactive Docs
Open http://localhost:8000/docs in your browser for a full interactive API explorer.

## 🔮 Future Enhancements

- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] ML-based semantic phrase matching
- [ ] User authentication & authorization
- [ ] Animation upload/management endpoints
- [ ] Caching layer (Redis)
- [ ] Rate limiting
- [ ] Logging & monitoring

## 📄 License

Part of the FYP D-VOICE project.
