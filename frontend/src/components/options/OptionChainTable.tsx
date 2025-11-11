import React from 'react';
import { OptionContractData } from '../../types/data';
import './OptionChainTable.css';

interface OptionChainTableProps {
  contracts: OptionContractData[];
  spotPrice: number;
}

const OptionChainTable: React.FC<OptionChainTableProps> = ({ contracts, spotPrice }) => {
  const calls = contracts.filter((c) => c.option_type === 'call');
  const puts = contracts.filter((c) => c.option_type === 'put');

  const formatNumber = (value: number | null | undefined, decimals: number = 2): string => {
    if (value === null || value === undefined) return '-';
    return value.toFixed(decimals);
  };

  const formatPercent = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return '-';
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="option-chain-table">
      <div className="spot-price">
        <strong>Spot Price:</strong> ${spotPrice.toFixed(2)}
      </div>

      <div className="chain-container">
        <div className="calls-section">
          <h3>Calls</h3>
          <table>
            <thead>
              <tr>
                <th>Strike</th>
                <th>Last</th>
                <th>Bid</th>
                <th>Ask</th>
                <th>Vol</th>
                <th>OI</th>
                <th>IV</th>
              </tr>
            </thead>
            <tbody>
              {calls.map((contract, idx) => (
                <tr
                  key={idx}
                  className={contract.strike <= spotPrice ? 'in-the-money' : ''}
                >
                  <td>{formatNumber(contract.strike, 0)}</td>
                  <td>{formatNumber(contract.last_price)}</td>
                  <td>{formatNumber(contract.bid)}</td>
                  <td>{formatNumber(contract.ask)}</td>
                  <td>{contract.volume ?? '-'}</td>
                  <td>{contract.open_interest ?? '-'}</td>
                  <td>{formatPercent(contract.implied_volatility)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="puts-section">
          <h3>Puts</h3>
          <table>
            <thead>
              <tr>
                <th>Strike</th>
                <th>Last</th>
                <th>Bid</th>
                <th>Ask</th>
                <th>Vol</th>
                <th>OI</th>
                <th>IV</th>
              </tr>
            </thead>
            <tbody>
              {puts.map((contract, idx) => (
                <tr
                  key={idx}
                  className={contract.strike >= spotPrice ? 'in-the-money' : ''}
                >
                  <td>{formatNumber(contract.strike, 0)}</td>
                  <td>{formatNumber(contract.last_price)}</td>
                  <td>{formatNumber(contract.bid)}</td>
                  <td>{formatNumber(contract.ask)}</td>
                  <td>{contract.volume ?? '-'}</td>
                  <td>{contract.open_interest ?? '-'}</td>
                  <td>{formatPercent(contract.implied_volatility)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default OptionChainTable;
