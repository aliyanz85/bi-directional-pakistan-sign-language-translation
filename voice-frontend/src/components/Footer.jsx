import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';

export const Footer = () => {
  const { language } = useLanguage();

  return (
    <footer className="bg-gray-900 text-gray-300 border-t border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* About */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-voice-green rounded-lg flex items-center justify-center">
                <span className="text-white font-bold">V</span>
              </div>
              <span className="font-bold text-lg text-white">VOICE</span>
            </div>
            <p className="text-sm text-gray-400">
              {language === 'en'
                ? 'Real-time translation between Pakistani Sign Language, Text, and Speech.'
                : 'پاکستانی سائن لینگویج، متن اور تقریر کے درمیان حقیقی وقت میں ترجمہ۔'}
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-white mb-4">
              {language === 'en' ? 'Quick Links' : 'فوری لنکس'}
            </h4>
            <ul className="space-y-2 text-sm">
              <li><a href="https://github.com/Najamhassan86/VOICE-FYP" target="_blank" rel="noopener noreferrer" className="hover:text-voice-green transition-colors">GitHub Repository</a></li>
              <li><a href="https://github.com/Najamhassan86/VOICE-FYP/blob/main/README.md" target="_blank" rel="noopener noreferrer" className="hover:text-voice-green transition-colors">Documentation</a></li>
              <li><a href="tel:+923171975525" className="hover:text-voice-green transition-colors">{language === 'en' ? 'Contact' : 'رابطہ'} (0317 1975525)</a></li>
              <li><a href="https://github.com/Najamhassan86/VOICE-FYP/blob/main/LICENSE" target="_blank" rel="noopener noreferrer" className="hover:text-voice-green transition-colors">{language === 'en' ? 'License' : 'لائسنس'}</a></li>
            </ul>
          </div>

          {/* Credits */}
          <div>
            <h4 className="font-semibold text-white mb-4">
              {language === 'en' ? 'Credits' : 'کریڈٹس'}
            </h4>
            <p className="text-sm text-gray-400 mb-2">
              <strong>NUCES:</strong> FAST National University of Computer and Emerging Sciences
            </p>
            <p className="text-sm text-gray-400">
              <strong>{language === 'en' ? 'Supervised by' : 'نگرانی کے تحت'}:</strong> Mr. Almas Khan
            </p>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-800 pt-8 text-center text-sm text-gray-400">
          <p>
            © 2024 VOICE - F25-385-D. {language === 'en' ? 'All rights reserved.' : 'تمام حقوق محفوظ ہیں۔'}
          </p>
          <p className="mt-2">
            {language === 'en'
              ? 'Final Year Project - Session 2022-2026'
              : 'فائنل ایئر پروجیکٹ - سیشن 2022-2026'}
          </p>
        </div>
      </div>
    </footer>
  );
};
