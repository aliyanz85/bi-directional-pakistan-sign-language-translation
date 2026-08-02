import React, { createContext, useContext, useState } from 'react';

const LanguageContext = createContext();

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within LanguageProvider');
  }
  return context;
};

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState('en'); // 'en' or 'ur'

  const translations = {
    en: {
      // Navigation
      nav_home: 'Home',
      nav_psl_to_text: 'PSL to Text',
      nav_text_to_psl: 'Text to PSL',
      nav_learn: 'Learn',
      nav_dashboard: 'Dashboard',

      // Home
      home_title: 'VOICE',
      home_subtitle: 'Bridging Communication Gaps',
      home_tagline: 'Real-time translation between Pakistani Sign Language, Text, and Speech',
      home_start: 'Start Translation',
      home_about: 'About',
      home_team: 'Meet the Team',
      home_desc1: 'Connect deaf and hearing communities through intelligent sign language recognition.',
      home_desc2: 'Experience seamless real-time translation powered by AI.',
      home_features: 'Features',
      home_feature1: 'Real-time PSL Recognition',
      home_feature2: 'Bilingual Output (Urdu/English)',
      home_feature3: '3D Avatar Animation',
      home_feature4: 'Interactive Learning Module',

      // PSL to Text
      psl_title: 'PSL to Text & Speech',
      psl_subtitle: 'Perform sign language gestures to translate',
      psl_start_capture: 'Start Capture',
      psl_stop_capture: 'Stop Capture',
      psl_waiting: 'Waiting for gesture...',
      psl_recognized: 'Recognized Sign',
      psl_confidence: 'Confidence',
      psl_fps: 'FPS',
      psl_latency: 'Latency',
      psl_play_audio: 'Play Audio',
      psl_permissions: 'Camera permissions denied',
      psl_low_confidence: 'Confidence too low',

      // Text to PSL
      text_title: 'Text to PSL',
      text_subtitle: 'Type text to see animated sign language',
      text_language: 'Language',
      text_input_placeholder: 'Enter text here...',
      text_translate: 'Translate to PSL',
      text_replay: 'Replay Animation',
      text_avatar_demo: 'Avatar Animation (Demo)',

      // Learning
      learn_title: 'Learn PSL',
      learn_subtitle: 'Practice and get feedback on your signs',
      learn_record: 'Record Sign',
      learn_submit: 'Submit for Feedback',
      learn_similarity: 'Similarity Score',
      learn_reference: 'Reference Sign',
      learn_your_attempt: 'Your Attempt',
      learn_feedback: 'Feedback',
      learn_hints: 'Hints for improvement',

      // Dashboard
      dash_title: 'Analytics Dashboard',
      dash_accuracy: 'Model Accuracy',
      dash_latency: 'Average Latency (ms)',
      dash_fps: 'FPS Trend',
      dash_sessions: 'Total Sessions',

      // Common
      loading: 'Loading...',
      error: 'Error',
      success: 'Success',
      try_again: 'Try Again',
      back: 'Back',
      close: 'Close',
      language_select: 'Select Language',
      english: 'English',
      urdu: 'اردو',
    },
    ur: {
      // Navigation
      nav_home: 'ہوم',
      nav_psl_to_text: 'PSL سے متن',
      nav_text_to_psl: 'متن سے PSL',
      nav_learn: 'سیکھیں',
      nav_dashboard: 'ڈیش بورڈ',

      // Home
      home_title: 'VOICE',
      home_subtitle: 'رابطے کے فاصلوں کو ختم کریں',
      home_tagline: 'پاکستانی سائن لینگویج، متن اور تقریر کے درمیان حقیقی وقت میں ترجمہ',
      home_start: 'ترجمہ شروع کریں',
      home_about: 'تعارف',
      home_team: 'ٹیم سے ملیں',
      home_desc1: 'بہرے اور سننے والے کمیونٹیز کو ذہین سائن لینگویج کی شناخت کے ذریعے جوڑیں۔',
      home_desc2: 'AI کی طاقت سے حقیقی وقت میں ترجمہ کا تجربہ کریں۔',
      home_features: 'خصوصیات',
      home_feature1: 'حقیقی وقت میں PSL کی شناخت',
      home_feature2: 'دو لسانی آؤٹ پٹ (اردو/انگریزی)',
      home_feature3: '3D اوتار حرکت',
      home_feature4: 'متحرک سیکھنے کا ماڈیول',

      // PSL to Text
      psl_title: 'PSL سے متن اور تقریر',
      psl_subtitle: 'سائن لینگویج کے اشارے کریں ترجمہ کے لیے',
      psl_start_capture: 'کیپچر شروع کریں',
      psl_stop_capture: 'کیپچر بند کریں',
      psl_waiting: 'اشارے کا انتظار...',
      psl_recognized: 'شناخت شدہ سائن',
      psl_confidence: 'اعتماد',
      psl_fps: 'FPS',
      psl_latency: 'تاخیر',
      psl_play_audio: 'آڈیو چلائیں',
      psl_permissions: 'کیمرے کی اجازت مسترد ہو گئی',
      psl_low_confidence: 'اعتماد بہت کم ہے',

      // Text to PSL
      text_title: 'متن سے PSL',
      text_subtitle: 'اپنے متن کو متحرک سائن لینگویج میں دیکھیں',
      text_language: 'زبان',
      text_input_placeholder: 'یہاں متن درج کریں...',
      text_translate: 'PSL میں ترجمہ کریں',
      text_replay: 'دوبارہ چلائیں',
      text_avatar_demo: 'اوتار حرکت (ڈیمو)',

      // Learning
      learn_title: 'PSL سیکھیں',
      learn_subtitle: 'اپنے اشاروں کا مشق کریں اور رائے حاصل کریں',
      learn_record: 'سائن ریکارڈ کریں',
      learn_submit: 'رائے کے لیے جمع کریں',
      learn_similarity: 'مماثلت کا اسکور',
      learn_reference: 'حوالہ سائن',
      learn_your_attempt: 'آپ کی کوشش',
      learn_feedback: 'رائے',
      learn_hints: 'بہتری کے لیے تجاویز',

      // Dashboard
      dash_title: 'تجزیاتی ڈیش بورڈ',
      dash_accuracy: 'ماڈل کی درستگی',
      dash_latency: 'اوسط تاخیر (ms)',
      dash_fps: 'FPS رجحان',
      dash_sessions: 'کل سیشنز',

      // Common
      loading: 'لوڈ ہو رہا ہے...',
      error: 'خرابی',
      success: 'کامیاب',
      try_again: 'دوبارہ کوشش کریں',
      back: 'واپس',
      close: 'بند کریں',
      language_select: 'زبان منتخب کریں',
      english: 'English',
      urdu: 'اردو',
    }
  };

  const t = (key) => translations[language][key] || key;

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};
