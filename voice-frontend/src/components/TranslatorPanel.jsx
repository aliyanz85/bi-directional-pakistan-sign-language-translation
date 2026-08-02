import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Volume2, Copy, Check } from 'lucide-react';
import { useState } from 'react';

export const TranslatorPanel = ({ recognizedSign, confidence, hasAudio, onPlayAudio }) => {
  const { t } = useLanguage();
  const [copied, setCopied] = useState(false);

  const handleCopyToClipboard = () => {
    if (recognizedSign) {
      navigator.clipboard.writeText(recognizedSign);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const confidencePercentage = Math.round((confidence || 0) * 100);
  const confidenceColor =
    confidencePercentage >= 80 ? 'bg-green-500' :
    confidencePercentage >= 60 ? 'bg-yellow-500' :
    'bg-red-500';

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      {/* Recognized Sign */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider">
          {t('psl_recognized')}
        </h3>
        <div className="min-h-16 bg-voice-light rounded-lg p-4 flex items-center justify-center">
          {recognizedSign ? (
            <div className="text-center">
              <p className="text-3xl font-bold text-voice-green mb-2">{recognizedSign}</p>
              <p className="text-xs text-gray-500">Gesture detected</p>
            </div>
          ) : (
            <p className="text-gray-400 text-sm">{t('psl_waiting')}</p>
          )}
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider">
            {t('psl_confidence')}
          </h3>
          <span className="text-lg font-bold text-voice-green">
            {confidencePercentage}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full ${confidenceColor} transition-all duration-300`}
            style={{ width: `${confidencePercentage}%` }}
          />
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onPlayAudio}
          disabled={!hasAudio}
          className="flex-1 flex items-center justify-center gap-2 bg-voice-green text-white px-4 py-3 rounded-lg font-medium hover:bg-voice-green/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Volume2 className="w-5 h-5" />
          {t('psl_play_audio')}
        </button>
        <button
          onClick={handleCopyToClipboard}
          disabled={!recognizedSign}
          className="flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium border-2 border-voice-green text-voice-green hover:bg-voice-light transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {copied ? (
            <Check className="w-5 h-5" />
          ) : (
            <Copy className="w-5 h-5" />
          )}
        </button>
      </div>

      {/* Info Box */}
      {!recognizedSign && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm text-blue-700">
            💡 {t('psl_waiting')} Make a clear gesture in front of the camera.
          </p>
        </div>
      )}
    </div>
  );
};
