import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { AvatarPlayer } from '../components/AvatarPlayer';

export const TexttoPSL = () => {
  const { t, language } = useLanguage();
  const [text, setText] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [isAnimating, setIsAnimating] = useState(false);

  // Quick words for English
  const quickWordsEnglish = [
    'excited', 'good', 'smart', 'yes',
    'book', 'ready', 'quick', 'important',
    'fantastic', 'healthy', 'peaceful', 'new'
  ];

  // Quick words for Urdu
  const quickWordsUrdu = [
    'خوش',      // excited/happy
    'اچھا',     // good
    'ذہین',     // smart
    'ہاں',      // yes
    'کتاب',     // book
    'تیار',     // ready
    'تیز',      // quick
    'اہم',      // important
    'بہترین',   // fantastic
    'تندرست',   // healthy
    'پرامن',    // peaceful
    'نیا'       // new
  ];

  const handleTranslate = () => {
    if (text.trim()) {
      setIsAnimating(true);
    }
  };

  const handleAnimationStart = () => {
    // Animation started
  };

  const handleAnimationEnd = () => {
    setIsAnimating(false);
  };

  const handleClear = () => {
    setText('');
    setIsAnimating(false);
  };

  const handleQuickWord = (word) => {
    // Stop current animation and start new one immediately
    setIsAnimating(false);
    setText(word);
    // Use setTimeout to ensure state updates before triggering animation
    setTimeout(() => {
      setIsAnimating(true);
    }, 50);
  };

  return (
    <div className="min-h-screen bg-white py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-poppins text-4xl font-bold text-gray-900 mb-2">
            {t('text_title')}
          </h1>
          <p className="text-lg text-gray-600">
            {t('text_subtitle')}
          </p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Section */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-6 space-y-4">
              {/* Language Selection */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  {t('text_language')}
                </label>
                <select
                  value={selectedLanguage}
                  onChange={(e) => setSelectedLanguage(e.target.value)}
                  className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-voice-green focus:outline-none transition-colors"
                >
                  <option value="en">English</option>
                  <option value="ur">اردو</option>
                </select>
              </div>

              {/* Text Input */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  {language === 'en' ? 'Enter Text' : 'متن درج کریں'}
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={selectedLanguage === 'ur'
                    ? 'اردو میں متن درج کریں...'
                    : 'Enter text in English...'}
                  rows="6"
                  className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:border-voice-green focus:outline-none transition-colors resize-none"
                  dir={selectedLanguage === 'ur' ? 'rtl' : 'ltr'}
                />
                <p className="text-xs text-gray-500 mt-2">
                  {text.length} {language === 'en' ? 'characters' : 'حروف'} • {text.split(/\s+/).filter(w => w).length} {language === 'en' ? 'words' : 'الفاظ'}
                </p>
              </div>

              {/* Word Count Info */}
              <div className="bg-voice-light rounded-lg p-3">
                <p className="text-sm text-gray-700" dir={language === 'ur' ? 'rtl' : 'ltr'}>
                  💡 {selectedLanguage === 'ur'
                    ? 'اردو میں متن درج کریں۔ ہر لفظ اپنے اشاراتی اینیمیشن کے ساتھ ظاہر ہوگا۔'
                    : (language === 'en'
                      ? 'Enter text to animate. Each word will be displayed with its sign animation.'
                      : 'متن درج کریں۔ ہر لفظ اپنے سائن اینیمیشن کے ساتھ ظاہر ہوگا۔')}
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3">
                <button
                  onClick={handleTranslate}
                  disabled={!text.trim() || isAnimating}
                  className="flex-1 bg-voice-green text-white px-4 py-3 rounded-lg font-medium hover:bg-voice-green/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAnimating
                    ? 'Animating...'
                    : t('text_translate')}
                </button>
                <button
                  onClick={handleClear}
                  className="px-4 py-3 rounded-lg font-medium border-2 border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  {language === 'en' ? 'Clear' : 'صاف کریں'}
                </button>
              </div>
            </div>

            {/* Quick Words */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="font-semibold text-gray-900 mb-4 uppercase text-sm">
                {language === 'en' ? 'Quick Words' : 'فوری الفاظ'}
              </h3>
              <div className="grid grid-cols-2 gap-2">
                {(selectedLanguage === 'ur' ? quickWordsUrdu : quickWordsEnglish).map((word) => (
                  <button
                    key={word}
                    onClick={() => handleQuickWord(word)}
                    className={`p-2 text-sm rounded-lg border-2 border-voice-light hover:border-voice-green hover:bg-voice-light transition-colors text-left font-medium text-gray-700 ${selectedLanguage === 'en' ? 'capitalize' : ''}`}
                  >
                    {word}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Avatar Player Section */}
          <div className="lg:col-span-2">
            <AvatarPlayer
              text={text}
              language={selectedLanguage}
              isAnimating={isAnimating}
              onAnimationStart={handleAnimationStart}
              onAnimationEnd={handleAnimationEnd}
            />

            {/* Tips */}
            <div className="mt-6 bg-blue-50 border-l-4 border-blue-500 rounded-lg p-4 space-y-2">
              <h4 className="font-semibold text-blue-900" dir={language === 'ur' ? 'rtl' : 'ltr'}>
                💡 {language === 'en' ? 'Tips for Best Results' : 'بہترین نتائج کے لیے تجاویز'}
              </h4>
              <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside" dir={language === 'ur' ? 'rtl' : 'ltr'}>
                <li>{selectedLanguage === 'ur'
                  ? 'عام الفاظ استعمال کریں جو فہرست میں موجود ہیں'
                  : (language === 'en'
                    ? 'Use simple words from the available list'
                    : 'دستیاب فہرست سے سادہ الفاظ استعمال کریں')}</li>
                <li>{selectedLanguage === 'ur'
                  ? 'ایک وقت میں ایک لفظ بہتر کام کرتا ہے'
                  : (language === 'en'
                    ? 'One word at a time works best'
                    : 'ایک وقت میں ایک لفظ بہتر ہے')}</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Available Words Info */}
        <div className="mt-8 p-4 bg-green-50 border-l-4 border-green-500 rounded-lg">
          <p className="text-sm text-green-800 font-semibold mb-2">
            {language === 'en'
              ? '✅ 32 Sign Language Animations Available'
              : '✅ 32 اشاراتی زبان کی اینیمیشنز دستیاب ہیں'}
          </p>
          {selectedLanguage === 'ur' ? (
            <p className="text-sm text-green-700" dir="rtl">
              یہ الفاظ آزمائیں: الرٹ، کتاب، محتاط، سستا، پاگل، خطرناک، مہذب، احمق، خوش، انتہائی، بہترین، دور، خوفزدہ، غیر ملکی، مضحکہ خیز، اچھا، صحت مند، بھاری، اہم، عقلمند، دلچسپ، دیر، کم، نیا، نہیں، شور، پرامن، تیز، تیار، محفوظ، ذہین، ہاں
            </p>
          ) : (
            <p className="text-sm text-green-700">
              Try these words: alert, book, careful, cheap, crazy, dangerous, decent, dumb, excited, extreme, fantastic, far, fearful, foreign, funny, good, healthy, heavy, important, intelligent, interesting, late, less, new, no, noisy, peaceful, quick, ready, secure, smart, yes
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
