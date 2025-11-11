import React from 'react';
import Plot from 'react-plotly.js';
import { VolSurfacePoint } from '../../types/surface';

interface VolSurface3DProps {
  surfacePoints: VolSurfacePoint[];
  symbol: string;
}

const VolSurface3D: React.FC<VolSurface3DProps> = ({ surfacePoints, symbol }) => {
  // Prepare data for 3D surface plot
  const strikeSet = new Set(surfacePoints.map((p) => p.strike));
  const strikes = Array.from(strikeSet).sort((a, b) => a - b);

  const expirySet = new Set(surfacePoints.map((p) => p.expiry));
  const expiries = Array.from(expirySet).sort((a, b) => a - b);

  // Create z-matrix for surface
  const zMatrix: (number | null)[][] = expiries.map((expiry) =>
    strikes.map((strike) => {
      const point = surfacePoints.find(
        (p) => Math.abs(p.strike - strike) < 0.01 && Math.abs(p.expiry - expiry) < 0.001
      );
      return point ? point.implied_vol * 100 : null; // Convert to percentage
    })
  );

  return (
    <Plot
      data={[
        {
          type: 'surface',
          x: strikes,
          y: expiries,
          z: zMatrix,
          colorscale: 'Viridis',
          colorbar: {
            title: { text: 'IV %' },
          },
        } as any,
      ]}
      layout={{
        title: { text: `${symbol} Implied Volatility Surface` },
        autosize: true,
        scene: {
          xaxis: { title: { text: 'Strike' } },
          yaxis: { title: { text: 'Time to Expiry (years)' } },
          zaxis: { title: { text: 'Implied Volatility (%)' } },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.3 },
          },
        },
        margin: { l: 0, r: 0, b: 0, t: 40 },
      }}
      config={{
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      }}
      style={{ width: '100%', height: '600px' }}
    />
  );
};

export default VolSurface3D;
