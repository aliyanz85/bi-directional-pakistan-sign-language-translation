# D-VOICE Setup Guide

Complete setup instructions for the D-VOICE animation system (backend + frontend).

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **npm** or **yarn**
- **Git** (optional, for version control)

## 🚀 Quick Start (Development)

Follow these steps to get both backend and frontend running:

### Step 1: Setup Backend

```bash
# Navigate to backend folder
cd voice-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn app.main:app --reload --port 8000
```

**Verify backend is running:**
- Open http://localhost:8000/health in your browser
- You should see: `{"status": "ok", ...}`
- Visit http://localhost:8000/docs for interactive API docs

### Step 2: Setup Frontend

**Open a NEW terminal window** (keep backend running):

```bash
# Navigate to frontend folder
cd voice-frontend

# Install dependencies
npm install

# Run the frontend dev server
npm run dev
```

**Verify frontend is running:**
- Open http://localhost:5173 in your browser
- You should see the D-VOICE home page

### Step 3: Test the Integration

1. Navigate to **Text to PSL** page in the frontend
2. Enter a word like "**excited**" or "**hello**"
3. Click **Translate**
4. Watch the 3D avatar animation play!

## 📂 Project Structure

```
FYP D-VOICE/
├── voice-backend/              # FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI app
│   │   ├── animations_config.json  # Animation metadata
│   │   ├── api/
│   │   │   └── animations.py  # API endpoints
│   │   └── core/
│   │       └── config.py      # Configuration
│   ├── requirements.txt
│   └── README.md
│
├── voice-frontend/            # React frontend
│   ├── public/
│   │   └── animations/        # 32 GLB files
│   ├── src/
│   │   ├── api/
│   │   │   └── animationApi.js
│   │   ├── components/
│   │   │   ├── AnimationPlayer3D.jsx
│   │   │   └── AvatarPlayer.jsx
│   │   ├── pages/
│   │   │   └── TexttoPSL.jsx
│   │   └── ...
│   ├── .env
│   ├── package.json
│   └── README.md
│
├── animation-player/          # Reference implementation (standalone)
│   └── ...
│
└── SETUP_GUIDE.md            # This file
```

## 🔧 Configuration

### Backend Configuration

**File:** `voice-backend/app/core/config.py`

Key settings:
- `PORT`: Backend port (default: 8000)
- `CORS_ORIGINS`: Allowed frontend URLs

To change port, create `voice-backend/.env`:
```env
PORT=8001
```

### Frontend Configuration

**File:** `voice-frontend/.env`

```env
VITE_API_BASE_URL=http://localhost:8000
```

If backend runs on a different port, update this URL.

## 🧪 Testing the System

### Test Backend Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get all animations
curl http://localhost:8000/api/animations

# Get specific animation
curl http://localhost:8000/api/animations/excited

# Resolve phrase to animation
curl -X POST http://localhost:8000/api/resolve-animation \
  -H "Content-Type: application/json" \
  -d '{"phrase": "I am excited", "language": "psl"}'
```

### Test Frontend

1. **Text to PSL Translation**:
   - Go to http://localhost:5173/text-to-psl
   - Enter: "excited good smart"
   - Click "Translate"
   - Should play 3 animations in sequence

2. **Check Browser Console**:
   - Open DevTools (F12)
   - Should see successful API calls
   - No errors about missing animations

## 🐛 Common Issues

### Issue: Backend won't start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
# Make sure virtual environment is activated
cd voice-backend
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Frontend can't connect to backend

**Error:** "Failed to fetch animations" or CORS errors

**Solution:**
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check `.env` file has correct URL
3. Restart frontend: `npm run dev`
4. Check backend CORS config includes `http://localhost:5173`

### Issue: 3D animations not loading

**Error:** Blank screen or "Failed to load animation"

**Solution:**
1. Verify GLB files exist: `ls voice-frontend/public/animations/`
2. Should see 32 `.glb` files
3. If missing, copy from animation-player:
   ```bash
   cp animation-player/public/animations/*.glb voice-frontend/public/animations/
   ```
4. Clear browser cache and reload

### Issue: npm install fails

**Error:** Dependency resolution errors

**Solution:**
```bash
cd voice-frontend
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

## 📊 System Architecture

```
┌─────────────────┐
│  User Browser   │
│  (localhost:5173)│
└────────┬────────┘
         │
         │ HTTP Requests
         ▼
┌─────────────────────┐
│  React Frontend     │
│  - Text input       │
│  - 3D renderer      │
│  - API calls        │
└────────┬────────────┘
         │
         │ Axios HTTP Calls
         ▼
┌─────────────────────┐
│  FastAPI Backend    │
│  (localhost:8000)   │
│  - /api/animations  │
│  - /resolve-animation│
└────────┬────────────┘
         │
         │ Reads metadata
         ▼
┌─────────────────────┐
│ animations_config.json│
│ (Animation metadata)│
└─────────────────────┘

Frontend loads GLB files from:
/public/animations/*.glb
```

## 🔄 Data Flow

1. **User enters text** → "I am excited"
2. **Frontend splits words** → ["I", "am", "excited"]
3. **For each word**:
   - Frontend calls: `POST /api/resolve-animation` with word
   - Backend searches metadata for matching animation
   - Backend returns: `{ animation: {...}, matched_word: "excited", confidence: 1.0 }`
4. **Frontend loads GLB** → `/animations/excited.glb`
5. **AnimationPlayer3D renders** → Three.js renders 3D model
6. **Animation plays** → User sees sign language gesture
7. **Repeat** for next word

## 📦 Available Animations (32 total)

alert, book, careful, cheap, crazy, dangerous, decent, dumb, excited, extreme, fantastic, far, fearful, foreign, funny, good, healthy, heavy, important, intelligent, interesting, late, less, new, no, noisy, peaceful, quick, ready, secure, smart, yes

## 🎯 Next Steps

After setup, you can:

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Try different words**: Use words from the list above
3. **Modify animations**: Add new GLB files and update `animations_config.json`
4. **Customize frontend**: Edit components in `voice-frontend/src/`
5. **Add new features**: Both codebases are well-documented

## 📚 Further Reading

- **Backend README**: `voice-backend/README.md`
- **Frontend README**: `voice-frontend/README.md`
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Three Fiber**: https://docs.pmnd.rs/react-three-fiber

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting sections in individual READMEs
2. Verify all prerequisites are installed
3. Check browser console and terminal for error messages
4. Ensure both backend and frontend are running simultaneously

---

**Success Checklist:**

- [ ] Backend running on http://localhost:8000
- [ ] `/health` endpoint returns OK
- [ ] Frontend running on http://localhost:5173
- [ ] Can navigate to Text to PSL page
- [ ] Entering "excited" shows 3D animation
- [ ] No errors in browser console

If all boxes are checked, you're ready to go! 🎉
