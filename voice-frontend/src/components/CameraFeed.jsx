import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Loader } from 'lucide-react';

export const CameraFeed = ({
  videoRef,
  isActive,
  error,
  onStartCamera,
  onStopCamera,
  fps,
  isLoading = false
}) => {
  const { t } = useLanguage();

  return (
    <div className="w-full bg-gray-900 rounded-lg overflow-hidden shadow-lg">
      {/* Video Container */}
      <div className="relative w-full bg-black aspect-video flex items-center justify-center">
        {isActive ? (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            {/* FPS Counter */}
            <div className="absolute top-4 right-4 bg-black bg-opacity-50 px-3 py-2 rounded-lg text-white text-sm font-mono">
              <div>{fps} FPS</div>
            </div>
            {/* Recording Indicator */}
            <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 px-3 py-2 rounded-lg text-white text-sm font-medium animate-pulse-soft">
              <div className="w-2 h-2 bg-white rounded-full"></div>
              Recording
            </div>
            {/* Backend Integration Notice */}
            <div className="absolute bottom-4 left-4 bg-blue-900/90 px-3 py-2 rounded text-xs text-blue-100 max-w-xs border border-blue-700">
              💡 <strong>Backend Implementation:</strong> We will use Flask/FastAPI with ML model for predictions. See BACKEND_INTEGRATION.md
            </div>
          </>
        ) : (
          <div className="text-center">
            {isLoading ? (
              <>
                <Loader className="w-12 h-12 text-voice-green mx-auto mb-4 animate-spin" />
                <p className="text-white text-sm">{t('loading')}</p>
              </>
            ) : (
              <>
                <div className="w-16 h-16 bg-voice-green bg-opacity-20 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <svg
                    className="w-8 h-8 text-voice-green"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path d="M12.12 11.28a1.5 1.5 0 0 0-2.24 0l-.88.88a.75.75 0 0 0 0 1.06l.88.88a1.5 1.5 0 0 0 2.24 0l.88-.88a.75.75 0 0 0 0-1.06l-.88-.88ZM10 5a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z" />
                  </svg>
                </div>
                <p className="text-white text-lg font-medium">
                  {error ? t('psl_permissions') : t('psl_waiting')}
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="bg-white px-6 py-4 flex gap-4 flex-wrap">
        {!isActive ? (
          <button
            onClick={onStartCamera}
            disabled={isLoading}
            className="flex-1 min-w-[200px] bg-voice-green text-white px-6 py-3 rounded-lg font-medium hover:bg-voice-green/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {t('psl_start_capture')}
          </button>
        ) : (
          <button
            onClick={onStopCamera}
            className="flex-1 min-w-[200px] bg-red-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-red-600 transition-colors"
          >
            {t('psl_stop_capture')}
          </button>
        )}
      </div>

      {/* Hidden Canvas for Frame Capture */}
      <canvas
        ref={videoRef?.canvas || null}
        style={{ display: 'none' }}
      />
    </div>
  );
};
