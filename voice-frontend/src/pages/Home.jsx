import React from 'react';
import { Link } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';
import { ArrowRight, Zap, Users, Smile, BookOpen } from 'lucide-react';

export const Home = () => {
  const { t, language } = useLanguage();

  return (
    <div className="min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-b from-voice-light to-white pt-20 pb-32 sm:pt-32 sm:pb-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            {/* Logo */}
            <div className="flex justify-center mb-6">
              <div className="w-24 h-24 bg-voice-green rounded-2xl flex items-center justify-center shadow-lg transform hover:scale-105 transition-transform">
                <span className="text-white text-5xl font-bold">V</span>
              </div>
            </div>

            {/* Main Text */}
            <h1 className="font-poppins text-5xl sm:text-6xl font-bold text-gray-900 mb-6 tracking-tight">
              {t('home_title')}
            </h1>
            <h2 className="font-poppins text-3xl sm:text-4xl font-bold text-voice-green mb-6">
              {t('home_subtitle')}
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-12 leading-relaxed">
              {t('home_tagline')}
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
              <Link
                to="/psl-to-text"
                className="inline-flex items-center gap-2 bg-voice-green text-white px-8 py-4 rounded-lg font-semibold hover:bg-voice-green/90 transition-all transform hover:scale-105 shadow-lg"
              >
                {t('home_start')}
                <ArrowRight className="w-5 h-5" />
              </Link>
              <a
                href="#features"
                className="inline-flex items-center gap-2 border-2 border-voice-green text-voice-green px-8 py-4 rounded-lg font-semibold hover:bg-voice-light transition-all"
              >
                {t('home_features')}
              </a>
            </div>

            {/* Hero Illustration */}
            <div className="mx-auto max-w-2xl">
              <div className="relative bg-gradient-to-b from-blue-50 to-blue-100 rounded-2xl p-8 shadow-xl">
                <div className="flex items-center justify-center h-64 text-6xl">
                  🤝
                </div>
                <p className="text-center text-sm text-gray-600 mt-4">
                  {language === 'en'
                    ? 'Bridging communication gaps between deaf and hearing communities'
                    : 'بہرے اور سننے والے کمیونٹیز کے درمیان رابطے کے فاصلوں کو ختم کریں'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-voice-light">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="font-poppins text-4xl font-bold text-gray-900 mb-4">
              {t('home_features')}
            </h2>
            <p className="text-lg text-gray-600">
              {language === 'en'
                ? 'Comprehensive tools for seamless sign language translation'
                : 'سائن لینگویج ترجمہ کے لیے جامع ٹولز'}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Feature 1 */}
            <div className="bg-white rounded-lg p-8 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 mt-1">
                  <Zap className="w-8 h-8 text-voice-green" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg text-gray-900 mb-2">
                    {t('home_feature1')}
                  </h3>
                  <p className="text-gray-600">
                    {language === 'en'
                      ? 'Advanced ML model recognizes Pakistani Sign Language gestures in real-time with high accuracy.'
                      : 'اعلیٰ درستگی کے ساتھ حقیقی وقت میں پاکستانی سائن لینگویج کی شناخت۔'}
                  </p>
                </div>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="bg-white rounded-lg p-8 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 mt-1">
                  <Users className="w-8 h-8 text-voice-green" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg text-gray-900 mb-2">
                    {t('home_feature2')}
                  </h3>
                  <p className="text-gray-600">
                    {language === 'en'
                      ? 'Automatic translation and synthesis in both Urdu and English for wider accessibility.'
                      : 'اردو اور انگریزی دونوں میں خودکار ترجمہ اور ترکیب۔'}
                  </p>
                </div>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="bg-white rounded-lg p-8 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 mt-1">
                  <Smile className="w-8 h-8 text-voice-green" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg text-gray-900 mb-2">
                    {t('home_feature3')}
                  </h3>
                  <p className="text-gray-600">
                    {language === 'en'
                      ? 'Animated 3D avatar demonstrates sign language from text input for reverse translation.'
                      : 'متن سے سائن لینگویج کو متحرک کریں۔'}
                  </p>
                </div>
              </div>
            </div>

            {/* Feature 4 */}
            <div className="bg-white rounded-lg p-8 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 mt-1">
                  <BookOpen className="w-8 h-8 text-voice-green" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg text-gray-900 mb-2">
                    {t('home_feature4')}
                  </h3>
                  <p className="text-gray-600">
                    {language === 'en'
                      ? 'Practice mode with AI feedback helps users improve their sign language skills.'
                      : 'AI رائے کے ساتھ سیکھنے کا موڈ۔'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="font-poppins text-4xl font-bold text-gray-900 mb-6">
            {language === 'en'
              ? 'Ready to get started?'
              : 'شروع کرنے کے لیے تیار ہیں؟'}
          </h2>
          <p className="text-lg text-gray-600 mb-8">
            {language === 'en'
              ? 'Choose your translation mode and begin communicating without barriers.'
              : 'اپنا ترجمہ کا موڈ منتخب کریں اور بغیر کسی رکاوٹ کے ابلاغ شروع کریں۔'}
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/psl-to-text"
              className="px-8 py-4 bg-voice-green text-white rounded-lg font-semibold hover:bg-voice-green/90 transition-all transform hover:scale-105"
            >
              {t('nav_psl_to_text')}
            </Link>
            <Link
              to="/text-to-psl"
              className="px-8 py-4 border-2 border-voice-green text-voice-green rounded-lg font-semibold hover:bg-voice-light transition-all"
            >
              {t('nav_text_to_psl')}
            </Link>
          </div>
        </div>
      </section>

      {/* Resources Section */}
      <section className="py-16 bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="font-poppins text-3xl font-bold text-gray-900 mb-2">
              {language === 'en' ? 'Project Resources' : 'پروجیکٹ کے وسائل'}
            </h2>
            <p className="text-gray-600">
              {language === 'en' ? 'Access our documentation, source code, and get in touch' : 'ہماری دستاویزات، سورس کوڈ تک رسائی حاصل کریں'}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* GitHub */}
            <a
              href="https://github.com/Najamhassan86/VOICE-FYP"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gray-50 rounded-lg p-6 text-center hover:shadow-lg transition-shadow hover:bg-gray-100"
            >
              <div className="text-4xl mb-3">🐙</div>
              <h3 className="font-semibold text-gray-900 mb-2">GitHub</h3>
              <p className="text-sm text-gray-600">
                {language === 'en' ? 'View source code' : 'سورس کوڈ دیکھیں'}
              </p>
            </a>

            {/* Documentation */}
            <a
              href="https://github.com/Najamhassan86/VOICE-FYP/blob/main/README.md"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gray-50 rounded-lg p-6 text-center hover:shadow-lg transition-shadow hover:bg-gray-100"
            >
              <div className="text-4xl mb-3">📚</div>
              <h3 className="font-semibold text-gray-900 mb-2">{language === 'en' ? 'Documentation' : 'دستاویزات'}</h3>
              <p className="text-sm text-gray-600">
                {language === 'en' ? 'Read full docs' : 'مکمل دستاویزات پڑھیں'}
              </p>
            </a>

            {/* License */}
            <a
              href="https://github.com/Najamhassan86/VOICE-FYP/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-gray-50 rounded-lg p-6 text-center hover:shadow-lg transition-shadow hover:bg-gray-100"
            >
              <div className="text-4xl mb-3">⚖️</div>
              <h3 className="font-semibold text-gray-900 mb-2">{language === 'en' ? 'License' : 'لائسنس'}</h3>
              <p className="text-sm text-gray-600">
                {language === 'en' ? 'View license' : 'لائسنس دیکھیں'}
              </p>
            </a>

            {/* Contact */}
            <a
              href="tel:+923171975525"
              className="bg-voice-light rounded-lg p-6 text-center hover:shadow-lg transition-shadow hover:bg-voice-green/10"
            >
              <div className="text-4xl mb-3">📞</div>
              <h3 className="font-semibold text-gray-900 mb-2">{language === 'en' ? 'Contact' : 'رابطہ'}</h3>
              <p className="text-sm text-voice-green font-semibold">0317 1975525</p>
            </a>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 bg-voice-light">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="font-poppins text-4xl font-bold text-gray-900 mb-4">
              {t('home_team')}
            </h2>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
              <div className="text-center">
                <div className="w-20 h-20 bg-voice-light rounded-full mx-auto mb-4 flex items-center justify-center text-3xl">
                  👨‍💼
                </div>
                <h3 className="font-semibold text-lg">Shaheer Zaman</h3>
                <p className="text-gray-600">22I-0805</p>
              </div>
              <div className="text-center">
                <div className="w-20 h-20 bg-voice-light rounded-full mx-auto mb-4 flex items-center justify-center text-3xl">
                  👨‍💼
                </div>
                <h3 className="font-semibold text-lg">Najam Hassan</h3>
                <p className="text-gray-600">22I-1332</p>
              </div>
            </div>

            <div className="text-center border-t pt-8">
              <div className="w-20 h-20 bg-voice-light rounded-full mx-auto mb-4 flex items-center justify-center text-3xl">
                👨‍💼
              </div>
              <h3 className="font-semibold text-lg mb-2">Aliyan Zafar</h3>
              <p className="text-gray-600 mb-6">20I-2414</p>

              <div className="text-sm text-gray-600 space-y-1">
                <p><strong>{language === 'en' ? 'Supervisor' : 'نگرانی کے تحت'}:</strong> Mr. Almas Khan</p>
                <p><strong>Department:</strong> Computer Science</p>
                <p><strong>University:</strong> FAST NUCES, Islamabad</p>
                <p><strong>Session:</strong> 2022-2026</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
