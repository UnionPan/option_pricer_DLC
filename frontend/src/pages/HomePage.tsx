import React, { useEffect, useState } from 'react';
import { marketApi } from '../services/marketApi';
import { MarketOverviewResponse, IndexChartsResponse } from '../types/market';
import MarketIndices from '../components/market/MarketIndices';
import IndexCharts from '../components/market/IndexCharts';
import Magnificent7 from '../components/market/Magnificent7';
import CommodityFutures from '../components/market/CommodityFutures';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import './HomePage.css';

const HomePage: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketOverviewResponse | null>(null);
  const [chartData, setChartData] = useState<IndexChartsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        setLoading(true);
        setError(null);

        const [overview, charts] = await Promise.all([
          marketApi.getMarketOverview(),
          marketApi.getIndexCharts(),
        ]);

        setMarketData(overview);
        setChartData(charts);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch market data');
        console.error('Error fetching market data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMarketData();

    // Refresh data every 1 hour
    const interval = setInterval(fetchMarketData, 3600000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="home-page">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="home-page">
        <ErrorMessage message={error} />
      </div>
    );
  }

  return (
    <div className="home-page">
      <div className="market-header">
        <h1>Market Overview</h1>
      </div>

      {/* Top Section: Indices (Left) + Charts (Right) */}
      <div className="market-top-section">
        <div className="market-top-left">
          {marketData && <MarketIndices indices={marketData.indices} />}
        </div>
        <div className="market-top-right">
          {chartData && <IndexCharts charts={chartData.charts} />}
        </div>
      </div>

      {/* Middle Section: Magnificent 7 */}
      <div className="market-middle-section">
        {marketData && <Magnificent7 stocks={marketData.magnificent7} />}
      </div>

      {/* Bottom Section: Commodity Futures */}
      <div className="market-bottom-section">
        {marketData && <CommodityFutures commodities={marketData.commodities} />}
      </div>
    </div>
  );
};

export default HomePage;
