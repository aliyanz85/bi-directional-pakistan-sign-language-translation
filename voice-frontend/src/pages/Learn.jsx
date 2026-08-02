import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useCamera } from '../hooks/useCamera';
import { Play, RotateCcw } from 'lucide-react';

export const Learn = () => {
  const { t, language } = useLanguage();
  const camera = useCamera();
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [feedback, setFeedback] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleStartRecording = async () => {
    setIsLoading(true);
    await camera.startCamera();
    setIsRecording(true);
    setRecordingTime(0);
    setFeedback(null);
    setIsLoading(false);
  };

  const handleStopRecording = () => {
    camera.stopCamera();
    setIsRecording(false);

    // Simulate feedback
    setTimeout(() => {
      setFeedback({
        similarity: Math.round(Math.random() * 40 + 60),
        suggestions: [
          language === 'en'
            ? 'Raise your hand slightly higher'
            : 'اپنا ہاتھ تھوڑا اوپر اٹھائیں',
          language === 'en'
            ? 'Make the movement more deliberate'
            : 'حرکت کو زیادہ واضح بنائیں',
          language === 'en'
            ? 'Keep your fingers together'
            : 'اپنی انگلیوں کو ایک ساتھ رکھیں'
        ]
      });
    }, 1500);
  };

  const handleResetFeedback = () => {
    setFeedback(null);
    setRecordingTime(0);
  };

  // Simulate recording time
  React.useEffect(() => {
    if (!isRecording) return;

    const timer = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [isRecording]);

  const referenceSigns = [
    { word: 'Hello', emoji: '👋' },
    { word: 'Thank You', emoji: '🙏' },
    { word: 'Help', emoji: '🆘' },
    { word: 'Water', emoji: '💧' },
    { word: 'Food', emoji: '🍽️' },
    { word: 'Happy', emoji: '😊' },
  ];

  return (
    <div className="min-h-screen bg-white py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-poppins text-4xl font-bold text-gray-900 mb-2">
            {t('learn_title')}
          </h1>
          <p className="text-lg text-gray-600">
            {t('learn_subtitle')}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Learning Mode */}
          <div className="lg:col-span-2 space-y-6">
            {/* Camera Section */}
            <div className="bg-gray-900 rounded-lg overflow-hidden shadow-lg">
              <div className="relative w-full bg-black aspect-video flex items-center justify-center">
                {camera.isActive ? (
                  <>
                    <video
                      ref={camera.videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-full object-cover"
                    />
                    {/* Recording Indicator */}
                    {isRecording && (
                      <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 px-3 py-2 rounded-lg text-white text-sm font-medium animate-pulse-soft">
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                        {t('learn_record')} ({recordingTime}s)
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center">
                    <div className="w-16 h-16 bg-voice-green bg-opacity-20 rounded-full mx-auto mb-4 flex items-center justify-center">
                      <svg
                        className="w-8 h-8 text-voice-green"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path d="M4 4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4z" />
                      </svg>
                    </div>
                    <p className="text-white text-lg font-medium">
                      {language === 'en' ? 'Ready to practice?' : 'سیکھنے کے لیے تیار؟'}
                    </p>
                  </div>
                )}
              </div>

              {/* Camera Controls */}
              <div className="bg-white px-6 py-4 flex gap-4">
                {!camera.isActive ? (
                  <button
                    onClick={handleStartRecording}
                    disabled={isLoading}
                    className="flex-1 bg-voice-green text-white px-6 py-3 rounded-lg font-medium hover:bg-voice-green/90 transition-colors disabled:opacity-50"
                  >
                    {t('learn_record')}
                  </button>
                ) : (
                  <button
                    onClick={handleStopRecording}
                    className="flex-1 bg-red-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-red-600 transition-colors"
                  >
                    {language === 'en' ? 'Stop Recording' : 'ریکارڈنگ روکیں'}
                  </button>
                )}
              </div>
            </div>

            {/* Side-by-Side Comparison */}
            {feedback && (
              <div className="bg-white rounded-lg shadow-lg p-6 space-y-4">
                <h3 className="font-semibold text-lg text-gray-900">
                  {language === 'en' ? 'Feedback & Comparison' : 'رائے اور موازنہ'}
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  {/* Reference */}
                  <div className="bg-voice-light rounded-lg p-4 text-center">
                    <p className="text-xs text-gray-600 uppercase font-semibold mb-3">
                      {t('learn_reference')}
                    </p>
                    <div className="text-5xl mb-3">👋</div>
                    <p className="font-semibold text-gray-900">Hello</p>
                  </div>

                  {/* User Attempt */}
                  <div className="bg-yellow-50 rounded-lg p-4 text-center border-2 border-yellow-200">
                    <p className="text-xs text-gray-600 uppercase font-semibold mb-3">
                      {t('learn_your_attempt')}
                    </p>
                    <div className="text-5xl mb-3">👋</div>
                    <p className="font-semibold text-gray-900">Your Sign</p>
                  </div>
                </div>

                {/* Similarity Score */}
                <div className="bg-white rounded-lg border-2 border-gray-200 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-gray-900">
                      {t('learn_similarity')}
                    </h4>
                    <span className="text-3xl font-bold text-voice-green">
                      {feedback.similarity}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-voice-green h-3 rounded-full transition-all duration-300"
                      style={{ width: `${feedback.similarity}%` }}
                    />
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    {feedback.similarity >= 80
                      ? '🎉 Excellent! You got it almost perfect!'
                      : feedback.similarity >= 60
                      ? '👍 Good effort! Keep practicing!'
                      : '💪 Keep trying! You\'ll get better!'}
                  </p>
                </div>

                {/* Suggestions */}
                <div className="bg-blue-50 rounded-lg p-4 border-l-4 border-blue-500">
                  <h4 className="font-semibold text-blue-900 mb-3">
                    {t('learn_hints')}
                  </h4>
                  <ul className="space-y-2">
                    {feedback.suggestions.map((suggestion, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-blue-800 text-sm">
                        <span className="inline-block w-5 h-5 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                          {idx + 1}
                        </span>
                        <span>{suggestion}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3">
                  <button
                    onClick={handleResetFeedback}
                    className="flex-1 flex items-center justify-center gap-2 bg-voice-green text-white px-4 py-3 rounded-lg font-medium hover:bg-voice-green/90 transition-colors"
                  >
                    <RotateCcw className="w-5 h-5" />
                    {language === 'en' ? 'Try Again' : 'دوبارہ کوشش کریں'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Practice Words */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="font-semibold text-gray-900 mb-4 uppercase text-sm">
                {language === 'en' ? 'Practice Words' : 'سیکھنے کے الفاظ'}
              </h3>
              <div className="space-y-3">
                {referenceSigns.map((sign) => (
                  <div
                    key={sign.word}
                    className="p-4 rounded-lg bg-voice-light border-2 border-transparent hover:border-voice-green transition-all cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-3xl">{sign.emoji}</span>
                      <span className="font-semibold text-gray-900">
                        {sign.word}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Progress Stats */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="font-semibold text-gray-900 mb-4 uppercase text-sm">
                {language === 'en' ? 'Your Progress' : 'آپ کی ترقی'}
              </h3>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-600 mb-1">
                    {language === 'en' ? 'Signs Practiced' : 'سیکھے گئے اشارے'}
                  </p>
                  <p className="text-3xl font-bold text-voice-green">0</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">
                    {language === 'en' ? 'Average Score' : 'اوسط اسکور'}
                  </p>
                  <p className="text-3xl font-bold text-voice-green">0%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">
                    {language === 'en' ? 'Total Sessions' : 'کل سیشنز'}
                  </p>
                  <p className="text-3xl font-bold text-voice-green">0</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Backend Integration Notice */}
        <div className="mt-8 p-4 bg-orange-50 border-l-4 border-orange-500 rounded-lg">
          <p className="text-sm text-orange-800 font-semibold mb-2">
             Backend Implementation required
          </p>
          <ul className="text-sm text-orange-700 list-disc list-inside space-y-1">
          </ul>
        </div>
      </div>
    </div>
  );
};
