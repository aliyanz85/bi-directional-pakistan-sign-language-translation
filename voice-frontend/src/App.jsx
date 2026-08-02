import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { LanguageProvider } from './contexts/LanguageContext';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Home } from './pages/Home';
import { PSLtoText } from './pages/PSLtoText';
import { TexttoPSL } from './pages/TexttoPSL';
import { Learn } from './pages/Learn';

function App() {
  return (
    <LanguageProvider>
      <Router>
        <div className="flex flex-col min-h-screen bg-white">
          <Navbar />
          <main className="flex-grow">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/psl-to-text" element={<PSLtoText />} />
              <Route path="/text-to-psl" element={<TexttoPSL />} />
              <Route path="/learn" element={<Learn />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </Router>
    </LanguageProvider>
  );
}

export default App;
