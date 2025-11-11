import React from 'react';
import './HomePage.css';

const HomePage: React.FC = () => {
  return (
    <div className="home-page">
      <header className="hero">
        <h1>Options Desk</h1>
        <p className="subtitle">
          A comprehensive options pricing, risk management, and hedging framework
        </p>
      </header>

      <div className="features">
        <div className="feature-card">
          <h2>Option Chains</h2>
          <p>Fetch and analyze real-time option chain data with strikes, prices, and Greeks.</p>
        </div>

        <div className="feature-card">
          <h2>Pricing Calculator</h2>
          <p>Calculate option prices and Greeks using Black-Scholes model.</p>
        </div>

        <div className="feature-card">
          <h2>Volatility Surface</h2>
          <p>Visualize implied volatility surfaces in 3D across strikes and maturities.</p>
        </div>
      </div>

      <div className="getting-started">
        <h2>Getting Started</h2>
        <p>Select a feature from the navigation above to begin exploring options analytics.</p>
      </div>
    </div>
  );
};

export default HomePage;
