import React, { useState } from 'react';
import './App.css';
import HomePage from './pages/HomePage';
import OptionChainPage from './pages/OptionChainPage';
import VolSurfacePage from './pages/VolSurfacePage';

type Page = 'market' | 'chain' | 'implied-vol' | 'greeks' | 'hedging' | 'approx-is-hedging' | 'portfolio-risk';

interface MenuItem {
  id: Page;
  label: string;
}

const menuItems: MenuItem[] = [
  { id: 'market', label: 'Market' },
  { id: 'chain', label: 'Chain & Price' },
  { id: 'implied-vol', label: 'Implied Vol' },
  { id: 'greeks', label: 'Greeks' },
  { id: 'hedging', label: 'Hedging' },
  { id: 'approx-is-hedging', label: 'Approx IS Hedging' },
  { id: 'portfolio-risk', label: 'Portfolio Risk' },
];

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('market');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handlePageChange = (page: Page) => {
    setCurrentPage(page);
    setMobileMenuOpen(false); // Close mobile menu when page changes
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'market':
        return <HomePage />;
      case 'chain':
        return <OptionChainPage />;
      case 'implied-vol':
        return <VolSurfacePage />;
      case 'greeks':
        return <div className="page-placeholder">Greeks Page - Coming Soon</div>;
      case 'hedging':
        return <div className="page-placeholder">Hedging Page - Coming Soon</div>;
      case 'approx-is-hedging':
        return <div className="page-placeholder">Approx IS Hedging Page - Coming Soon</div>;
      case 'portfolio-risk':
        return <div className="page-placeholder">Portfolio Risk Page - Coming Soon</div>;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="App">
      {/* Mobile Menu Button */}
      <button className="mobile-menu-button" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
        <div className="hamburger">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </button>

      {/* Sidebar - Desktop */}
      <aside className="sidebar">
        <div className="sidebar-header" onClick={() => handlePageChange('market')} style={{ cursor: 'pointer' }}>
          <h1 className="sidebar-title">AIS opt</h1>
        </div>
        <nav className="sidebar-nav">
          {menuItems.map((item) => (
            <button
              key={item.id}
              className={`sidebar-nav-item ${currentPage === item.id ? 'active' : ''}`}
              onClick={() => handlePageChange(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <>
          <div className="mobile-menu-overlay" onClick={() => setMobileMenuOpen(false)}></div>
          <div className="mobile-menu">
            <div className="mobile-menu-header">
              <h1 className="sidebar-title" onClick={() => handlePageChange('market')}>AIS opt</h1>
              <button className="close-button" onClick={() => setMobileMenuOpen(false)}>×</button>
            </div>
            <nav className="mobile-nav">
              {menuItems.map((item) => (
                <button
                  key={item.id}
                  className={`mobile-nav-item ${currentPage === item.id ? 'active' : ''}`}
                  onClick={() => handlePageChange(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </>
      )}

      <main className="main-content">{renderPage()}</main>
    </div>
  );
}

export default App;
