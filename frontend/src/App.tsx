import React, { useState } from 'react';
import './App.css';
import HomePage from './pages/HomePage';
import OptionChainPage from './pages/OptionChainPage';
import VolSurfacePage from './pages/VolSurfacePage';

type Page = 'home' | 'chain' | 'surface';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'chain':
        return <OptionChainPage />;
      case 'surface':
        return <VolSurfacePage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-brand">Options Desk</div>
        <div className="nav-links">
          <button
            className={currentPage === 'home' ? 'active' : ''}
            onClick={() => setCurrentPage('home')}
          >
            Home
          </button>
          <button
            className={currentPage === 'chain' ? 'active' : ''}
            onClick={() => setCurrentPage('chain')}
          >
            Option Chain
          </button>
          <button
            className={currentPage === 'surface' ? 'active' : ''}
            onClick={() => setCurrentPage('surface')}
          >
            Vol Surface
          </button>
        </div>
      </nav>
      <main className="main-content">{renderPage()}</main>
    </div>
  );
}

export default App;
