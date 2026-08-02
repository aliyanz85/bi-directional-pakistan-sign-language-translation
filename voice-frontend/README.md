# D-VOICE Frontend

React-based frontend application for the D-VOICE sign language translation platform. Features real-time 3D PSL (Pakistan Sign Language) animation playback powered by a FastAPI backend.

## 🚀 Features

- **Text to PSL Translation**: Enter text and see 3D sign language animations
- **PSL to Text Recognition**: Camera-based sign language recognition (coming soon)
- **3D Avatar Player**: Real-time 3D animation playback using Three.js
- **Multi-language Support**: English and Urdu interface
- **Responsive Design**: Built with Tailwind CSS
- **Backend Integration**: Connects to FastAPI backend for animation metadata

## 📁 Project Structure

```
voice-frontend/
├── public/
│   ├── animations/          # 32 GLB 3D animation files
│   └── index.html
├── src/
│   ├── api/
│   │   └── animationApi.js  # Backend API service layer
│   ├── components/
│   │   ├── AnimationPlayer3D.jsx   # 3D animation renderer
│   │   ├── AvatarPlayer.jsx        # Main avatar player component
│   │   ├── CameraFeed.jsx
│   │   ├── Footer.jsx
│   │   ├── Navbar.jsx
│   │   └── TranslatorPanel.jsx
│   ├── contexts/
│   │   └── LanguageContext.jsx
│   ├── hooks/
│   │   ├── useCamera.js
│   │   └── useWebSocket.js
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── Learn.jsx
│   │   ├── PSLtoText.jsx
│   │   └── TexttoPSL.jsx      # Main text-to-sign page
│   ├── App.jsx
│   ├── index.css
│   └── main.jsx
├── .env                      # Environment variables
├── .env.example             # Example environment config
├── package.json
└── README.md                # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Backend API running (see `voice-backend/README.md`)

### 1. Install Dependencies

```bash
cd voice-frontend
npm install
```

This will install all dependencies including:
- React & React Router
- Tailwind CSS
- Axios for API calls
- Three.js & React Three Fiber for 3D rendering
- Lucide React for icons

### 2. Configure Environment

The `.env` file is already created with default values:

```env
VITE_API_BASE_URL=http://localhost:8000
```

If your backend runs on a different port, update this value.

### 3. Run Development Server

```bash
npm run dev
```

The app will start at: **http://localhost:5173**

### 4. Build for Production

```bash
npm run build
```

Output will be in the `dist/` folder.

## 🎯 How It Works

### Data Flow: Text → Animation

1. **User Input**: User enters text in the TexttoPSL page
2. **API Call**: Frontend calls `/api/resolve-animation` endpoint
3. **Backend Processing**: Backend matches words to animation metadata
4. **Animation Loading**: Frontend loads the GLB file from `/public/animations/`
5. **3D Rendering**: AnimationPlayer3D renders the 3D avatar using Three.js
6. **Playback**: Animation plays with controls (play/pause/speed)

### Key Components

#### AnimationPlayer3D.jsx
- Renders 3D GLB animations using React Three Fiber
- Uses `GLTFLoader` to load models
- `AnimationMixer` controls playback
- Orbit controls for camera interaction

#### AvatarPlayer.jsx
- Main avatar player component
- Fetches animation metadata from backend
- Manages word-by-word animation playback
- Shows progress and controls

#### animationApi.js
- Axios-based API service layer
- Functions for fetching animations
- Resolves phrases to animations
- Centralized error handling

## 📡 API Integration

The frontend calls these backend endpoints:

```javascript
import {
  getAnimations,           // GET /api/animations
  getAnimationById,        // GET /api/animations/:id
  resolveAnimationForPhrase // POST /api/resolve-animation
} from './api/animationApi';
```

### Example Usage

```jsx
import { resolveAnimationForPhrase } from '../api/animationApi';

const result = await resolveAnimationForPhrase("hello", "psl");
// Returns: { animation: {...}, matched_word: "hello", confidence: 1.0 }
```

## 🎨 Pages

### 1. Home (`/`)
Landing page with navigation to features.

### 2. Text to PSL (`/text-to-psl`)
Main feature - convert text to sign language animations.
- Text input area
- Language selection (English/Urdu)
- 3D avatar player
- Play/pause controls
- Word-by-word progress tracking

### 3. PSL to Text (`/psl-to-text`)
Camera-based sign recognition (placeholder for ML integration).

### 4. Learn (`/learn`)
Educational content about PSL (coming soon).

## 🔧 Configuration

### Tailwind Custom Colors

Defined in `tailwind.config.js`:
```js
colors: {
  'voice-green': '#10B981',
  'voice-light': '#F0FDF4',
}
```

### Environment Variables

- `VITE_API_BASE_URL`: Backend API base URL (default: `http://localhost:8000`)

## 🐛 Troubleshooting

### Backend Connection Issues

**Problem**: "Failed to fetch animations" error

**Solution**:
1. Verify backend is running at http://localhost:8000
2. Check `/health` endpoint: `curl http://localhost:8000/health`
3. Verify CORS is configured correctly in backend
4. Check `.env` file has correct `VITE_API_BASE_URL`

### 3D Animation Not Loading

**Problem**: Animation doesn't appear or shows error

**Solution**:
1. Check browser console for errors
2. Verify GLB files exist in `public/animations/`
3. Check file path in backend response
4. Ensure Three.js dependencies are installed: `npm install`

### Build Errors

**Problem**: Build fails with dependency errors

**Solution**:
```bash
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## 📦 Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
```

## 🔮 Future Enhancements

- [ ] Real-time camera-based PSL recognition
- [ ] Animation speed controls
- [ ] Animation library browser
- [ ] Phrase bookmarks/favorites
- [ ] User accounts & history
- [ ] Mobile app version
- [ ] Offline mode support

## 📚 Dependencies

### Production
- **React** (^18.2.0): UI framework
- **React Router** (^6.20.0): Routing
- **Axios** (^1.6.2): HTTP client
- **Three.js** (^0.155.0): 3D rendering
- **@react-three/fiber** (^8.14.0): React renderer for Three.js
- **Lucide React** (^0.294.0): Icons
- **Recharts** (^2.10.3): Charts (for analytics)

### Development
- **Vite** (^5.0.8): Build tool & dev server
- **Tailwind CSS** (^3.3.6): Styling
- **PostCSS** (^8.4.32): CSS processing

## 🔗 Related Projects

- **voice-backend**: FastAPI backend for animation metadata
- **animation-player**: Original standalone 3D animation viewer (reference implementation)

## 📄 License

Part of the FYP D-VOICE project.

---

**Need Help?** Check the backend README or open an issue in the repository.
