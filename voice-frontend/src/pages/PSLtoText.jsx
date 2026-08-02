import React, { useState, useRef, useCallback } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { useMediaPipe } from '../hooks/useMediaPipe';
import { usePSLRecognition } from '../hooks/usePSLRecognition';
import { Volume2, Trash2, RotateCcw, Video, VideoOff, Hand } from 'lucide-react';

export const PSLtoText = () => {
  const { t, language } = useLanguage();
  const videoRef = useRef(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // MediaPipe hand detection
  const mediaPipe = useMediaPipe(videoRef, {
    onFeatures: (features) => {
      // Add features to PSL recognition buffer
      recognition.addFrame(features);
    },
    drawLandmarks: true
  });

  // PSL recognition with auto-recognition
  const recognition = usePSLRecognition({
    sequenceLength: 60,
    confidenceThreshold: 0.6,
    cooldownMs: 1500,
    autoRecognize: true
  });

  // Camera control
  const handleStartCamera = useCallback(async () => {
    try {
      await mediaPipe.startDetection();
    } catch (err) {
      console.error('Failed to start camera:', err);
    }
  }, [mediaPipe]);

  const handleStopCamera = useCallback(() => {
    mediaPipe.stopDetection();
    recognition.resetBuffer();
  }, [mediaPipe, recognition]);

  // Text-to-Speech
  const handleSpeak = useCallback(() => {
    const text = recognition.getSentenceText();
    if (!text) return;

    setIsSpeaking(true);
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.lang = language === 'ur' ? 'ur-PK' : 'en-US';

    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, [recognition, language]);

  // Clear sentence
  const handleClearSentence = useCallback(() => {
    window.speechSynthesis.cancel();
    recognition.clearSentence();
    setIsSpeaking(false);
  }, [recognition]);

  // Reset buffer
  const handleResetBuffer = useCallback(() => {
    recognition.resetBuffer();
  }, [recognition]);

  return (
    <div className="min-h-screen bg-white py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-poppins text-4xl font-bold text-gray-900 mb-2">
            {language === 'en' ? 'PSL to Text & Speech' : 'PSL سے متن اور آواز'}
          </h1>
          <p className="text-lg text-gray-600">
            {language === 'en'
              ? 'Perform sign language and see it converted to text in real-time'
              : 'اشاراتی زبان دکھائیں اور اسے فوری طور پر متن میں تبدیل ہوتے دیکھیں'}
          </p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Camera Feed Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Video Display */}
            <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
              {/* Video Element */}
              <video
                ref={videoRef}
                className="absolute inset-0 w-full h-full object-cover transform -scale-x-100"
                autoPlay
                playsInline
                muted
              />

              {/* Landmarks Canvas Overlay */}
              <canvas
                ref={mediaPipe.canvasRef}
                className="absolute inset-0 w-full h-full transform -scale-x-100"
              />

              {/* Status Overlays */}
              {!mediaPipe.isDetecting && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/50">
                  <div className="text-center text-white">
                    <Video className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-semibold">
                      {language === 'en' ? 'Camera Inactive' : 'کیمرہ غیر فعال'}
                    </p>
                  </div>
                </div>
              )}

              {/* Hand Detection Indicator */}
              {mediaPipe.isDetecting && (
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/50 px-3 py-2 rounded-lg">
                  <Hand className={`w-5 h-5 ${mediaPipe.handsDetected > 0 ? 'text-green-400' : 'text-gray-400'}`} />
                  <span className="text-white text-sm font-medium">
                    {mediaPipe.handsDetected} {language === 'en' ? 'Hand(s)' : 'ہاتھ'}
                  </span>
                </div>
              )}

              {/* FPS Counter */}
              {mediaPipe.isDetecting && (
                <div className="absolute top-4 right-4 bg-black/50 px-3 py-2 rounded-lg">
                  <span className="text-white text-sm font-medium">
                    {mediaPipe.fps} FPS
                  </span>
                </div>
              )}

              {/* Buffer Progress */}
              {mediaPipe.isDetecting && (
                <div className="absolute bottom-4 left-4 right-4">
                  <div className="bg-black/50 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white text-xs font-medium">
                        {language === 'en' ? 'Buffer' : 'بفر'}: {recognition.bufferLength}/60
                      </span>
                      <span className="text-white text-xs">
                        {recognition.bufferProgress.toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-voice-green h-2 rounded-full transition-all duration-300"
                        style={{ width: `${recognition.bufferProgress}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Recognizing Indicator */}
              {recognition.isRecognizing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                  <div className="bg-voice-green/90 text-white px-6 py-3 rounded-full font-semibold flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    {language === 'en' ? 'Recognizing...' : 'شناخت کر رہے ہیں...'}
                  </div>
                </div>
              )}
            </div>

            {/* Camera Controls */}
            <div className="flex gap-3">
              {!mediaPipe.isDetecting ? (
                <button
                  onClick={handleStartCamera}
                  disabled={mediaPipe.isInitialized && mediaPipe.isDetecting}
                  className="flex-1 flex items-center justify-center gap-2 bg-voice-green text-white px-6 py-3 rounded-lg font-medium hover:bg-voice-green/90 transition-colors disabled:opacity-50"
                >
                  <Video className="w-5 h-5" />
                  {language === 'en' ? 'Start Camera' : 'کیمرہ شروع کریں'}
                </button>
              ) : (
                <button
                  onClick={handleStopCamera}
                  className="flex-1 flex items-center justify-center gap-2 bg-red-500 text-white px-6 py-3 rounded-lg font-medium hover:bg-red-600 transition-colors"
                >
                  <VideoOff className="w-5 h-5" />
                  {language === 'en' ? 'Stop Camera' : 'کیمرہ بند کریں'}
                </button>
              )}

              <button
                onClick={handleResetBuffer}
                disabled={!mediaPipe.isDetecting}
                className="px-6 py-3 rounded-lg font-medium border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                <RotateCcw className="w-5 h-5" />
                {language === 'en' ? 'Reset' : 'دوبارہ ترتیب'}
              </button>
            </div>

            {/* Instructions */}
            {!mediaPipe.isDetecting && (
              <div className="bg-blue-50 border-l-4 border-blue-500 rounded-lg p-4">
                <h3 className="font-semibold text-blue-900 mb-2">
                  💡 {language === 'en' ? 'How to Use' : 'استعمال کیسے کریں'}
                </h3>
                <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
                  <li>{language === 'en' ? 'Click "Start Camera" to begin' : 'شروع کرنے کے لیے "کیمرہ شروع کریں" پر کلک کریں'}</li>
                  <li>{language === 'en' ? 'Perform PSL signs clearly in front of the camera' : 'کیمرے کے سامنے واضح طور پر PSL اشارے کریں'}</li>
                  <li>{language === 'en' ? 'Hold each sign for 2 seconds' : 'ہر اشارہ 2 سیکنڈ تک رکھیں'}</li>
                  <li>{language === 'en' ? 'Recognized words will appear in the prediction panel' : 'پہچانے گئے الفاظ پیشین گوئی پینل میں ظاہر ہوں گے'}</li>
                </ul>
              </div>
            )}

            {/* Error Display */}
            {(mediaPipe.error || recognition.error) && (
              <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4">
                <p className="text-sm text-red-800 font-semibold">
                  ⚠️ Error: {mediaPipe.error || recognition.error}
                </p>
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Current Prediction */}
            <div className="bg-white rounded-lg shadow-lg p-6 space-y-4">
              <h3 className="font-semibold text-gray-900 uppercase text-sm">
                {language === 'en' ? 'Current Sign' : 'موجودہ اشارہ'}
              </h3>

              {recognition.currentPrediction ? (
                <>
                  <div className="text-center py-6">
                    <p className="text-4xl font-bold text-voice-green mb-2">
                      {recognition.currentPrediction.label}
                    </p>
                    <p className="text-sm text-gray-600">
                      {language === 'en' ? 'Confidence' : 'اعتماد'}:{' '}
                      {(recognition.currentPrediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Top Predictions */}
                  <div>
                    <h4 className="text-xs font-semibold text-gray-600 uppercase mb-2">
                      {language === 'en' ? 'Top Predictions' : 'اوپر کی پیشین گوئیاں'}
                    </h4>
                    <div className="space-y-2">
                      {recognition.currentPrediction.top_predictions?.slice(0, 3).map((pred, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <div className="flex-1 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-voice-green h-2 rounded-full transition-all"
                              style={{ width: `${pred.confidence * 100}%` }}
                            />
                          </div>
                          <span className="text-xs font-medium text-gray-700 w-20 text-right">
                            {pred.label}
                          </span>
                          <span className="text-xs text-gray-500 w-12 text-right">
                            {(pred.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <p className="text-sm">
                    {language === 'en' ? 'Waiting for recognition...' : 'شناخت کا انتظار...'}
                  </p>
                </div>
              )}
            </div>

            {/* Sentence Builder */}
            <div className="bg-white rounded-lg shadow-lg p-6 space-y-4">
              <h3 className="font-semibold text-gray-900 uppercase text-sm">
                {language === 'en' ? 'Sentence' : 'جملہ'}
              </h3>

              {/* Sentence Display */}
              <div className="min-h-[100px] p-4 bg-gray-50 rounded-lg border-2 border-gray-200">
                {recognition.sentence.length > 0 ? (
                  <p className="text-lg text-gray-900 leading-relaxed">
                    {recognition.sentence.join(' ')}
                  </p>
                ) : (
                  <p className="text-sm text-gray-400 italic">
                    {language === 'en' ? 'No words yet...' : 'ابھی تک کوئی لفظ نہیں...'}
                  </p>
                )}
              </div>

              {/* Word Count */}
              <div className="flex items-center justify-between text-xs text-gray-600">
                <span>
                  {recognition.sentence.length} {language === 'en' ? 'word(s)' : 'لفظ'}
                </span>
                <span>
                  {recognition.getSentenceText().length} {language === 'en' ? 'character(s)' : 'حروف'}
                </span>
              </div>

              {/* Sentence Controls */}
              <div className="flex gap-2">
                <button
                  onClick={handleSpeak}
                  disabled={recognition.sentence.length === 0 || isSpeaking}
                  className="flex-1 flex items-center justify-center gap-2 bg-voice-green text-white px-4 py-2 rounded-lg font-medium hover:bg-voice-green/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Volume2 className="w-4 h-4" />
                  {isSpeaking
                    ? (language === 'en' ? 'Speaking...' : 'بول رہے ہیں...')
                    : (language === 'en' ? 'Speak' : 'بولیں')}
                </button>

                <button
                  onClick={recognition.undoLastWord}
                  disabled={recognition.sentence.length === 0}
                  className="px-4 py-2 rounded-lg font-medium border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title={language === 'en' ? 'Undo last word' : 'آخری لفظ واپس کریں'}
                >
                  <RotateCcw className="w-4 h-4" />
                </button>

                <button
                  onClick={handleClearSentence}
                  disabled={recognition.sentence.length === 0}
                  className="px-4 py-2 rounded-lg font-medium border-2 border-red-300 text-red-700 hover:bg-red-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title={language === 'en' ? 'Clear sentence' : 'جملہ صاف کریں'}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Stats */}
            {mediaPipe.isDetecting && (
              <div className="bg-voice-light rounded-lg p-6 space-y-4">
                <h3 className="font-semibold text-gray-900 uppercase text-sm">
                  {language === 'en' ? 'Live Statistics' : 'براہ راست اعداد و شمار'}
                </h3>
                <div className="space-y-3">
                  <div>
                    <p className="text-xs text-gray-600 uppercase font-semibold mb-1">
                      {language === 'en' ? 'Buffer Status' : 'بفر کی حالت'}
                    </p>
                    <p className="text-2xl font-bold text-voice-green">
                      {recognition.bufferLength}/60
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600 uppercase font-semibold mb-1">
                      {language === 'en' ? 'Words Recognized' : 'پہچانے گئے الفاظ'}
                    </p>
                    <p className="text-lg font-semibold text-gray-900">
                      {recognition.sentence.length}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600 uppercase font-semibold mb-1">
                      {language === 'en' ? 'Hands Detected' : 'ہاتھ پائے گئے'}
                    </p>
                    <p className="text-lg font-semibold text-gray-900">
                      {mediaPipe.handsDetected}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Available Signs Info */}
        <div className="mt-8 p-4 bg-green-50 border-l-4 border-green-500 rounded-lg">
          <p className="text-sm text-green-800 font-semibold mb-2">
            ✅ {language === 'en' ? '32 PSL Signs Available for Recognition' : '32 PSL اشارے شناخت کے لیے دستیاب ہیں'}
          </p>
          <p className="text-sm text-green-700">
            {language === 'en'
              ? 'Available signs: alert, book, careful, cheap, crazy, dangerous, decent, dumb, excited, extreme, fantastic, far, fearful, foreign, funny, good, healthy, heavy, important, intelligent, interesting, late, less, new, no, noisy, peaceful, quick, ready, secure, smart, yes'
              : 'دستیاب اشارے: الرٹ، کتاب، محتاط، سستا، پاگل، خطرناک، مہذب، احمق، خوش، انتہائی، بہترین، دور، خوفزدہ، غیر ملکی، مضحکہ خیز، اچھا، صحت مند، بھاری، اہم، عقلمند، دلچسپ، دیر، کم، نیا، نہیں، شور، پرامن، تیز، تیار، محفوظ، ذہین، ہاں'}
          </p>
        </div>
      </div>
    </div>
  );
};
