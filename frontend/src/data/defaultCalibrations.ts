/**
 * Default calibrated parameters for each model.
 * Users can load these instead of uploading their own calibration results.
 */

export const DEFAULT_CALIBRATIONS = {
  gbm: {
    ticker: 'EXAMPLE',
    model: 'gbm',
    parameters: {
      mu: 0.10,      // 10% drift
      sigma: 0.25,   // 25% volatility
    },
    diagnostics: {
      logLikelihood: -150.5,
      aic: 304.0,
      bic: 310.2,
    },
    timestamp: new Date().toISOString(),
    method: 'mle',
    measure: 'P-measure',
  },

  heston: {
    ticker: 'MSFT',
    model: 'heston',
    parameters: {
      mu: 0.10,           // Drift (~10% annual return)
      kappa: 3.0,         // Mean reversion speed
      theta: 0.0625,      // Long-term variance (25% vol)
      xi: 0.6,            // Vol of vol
      v0: 0.0625,         // Initial variance (25% vol)
      rho: -0.7,          // Correlation (leverage effect, typical equity skew)
    },
    diagnostics: {
      logLikelihood: null,
      aic: null,
      bic: null,
      note: 'Realistic parameters for MSFT: 25% vol, Feller satisfied (2κθ=0.375 > ξ²=0.36)',
    },
    timestamp: new Date().toISOString(),
    method: 'calibrated',
    measure: 'P-measure',
  },

  ou: {
    ticker: 'EXAMPLE',
    model: 'ou',
    parameters: {
      mu: 100.0,     // Long-term mean
      kappa: 1.5,    // Mean reversion speed
      sigma: 12.0,   // Volatility
    },
    diagnostics: {
      logLikelihood: -160.3,
      aic: 326.6,
      bic: 335.1,
    },
    timestamp: new Date().toISOString(),
    method: 'mle',
    measure: 'P-measure',
  },

  merton_jump: {
    ticker: 'EXAMPLE',
    model: 'merton_jump',
    parameters: {
      mu: 0.12,        // Drift
      sigma: 0.20,     // Diffusion volatility
      lambda: 0.5,     // Jump intensity (0.5 jumps/year)
      mu_j: -0.05,     // Average jump size (-5%)
      sigma_j: 0.10,   // Jump size volatility
    },
    diagnostics: {
      logLikelihood: -142.8,
      aic: 295.6,
      bic: 310.2,
    },
    timestamp: new Date().toISOString(),
    method: 'mle',
    measure: 'P-measure',
  },

  rough_bergomi: {
    ticker: 'EXAMPLE',
    model: 'rough_bergomi',
    parameters: {
      H: 0.1,         // Hurst parameter (roughness)
      eta: 1.9,       // Vol of vol
      rho: -0.75,     // Correlation
      xi0: 0.04,      // Initial forward variance
    },
    diagnostics: {
      logLikelihood: -138.5,
      aic: 285.0,
      bic: 298.4,
      variogram_r2: 0.92,
    },
    timestamp: new Date().toISOString(),
    method: 'moment_matching',
    measure: 'P-measure',
  },
};

export const MODEL_DESCRIPTIONS = {
  gbm: {
    name: 'Geometric Brownian Motion',
    description: 'Classic Black-Scholes model with constant volatility',
    useCases: 'Simple scenarios, educational purposes',
  },
  heston: {
    name: 'Heston Stochastic Volatility',
    description: 'Two-factor model with mean-reverting stochastic variance',
    useCases: 'Volatility smile, realistic vol dynamics',
  },
  ou: {
    name: 'Ornstein-Uhlenbeck',
    description: 'Mean-reverting process for commodities and interest rates',
    useCases: 'Mean-reverting assets, spreads',
  },
  merton_jump: {
    name: 'Merton Jump-Diffusion',
    description: 'GBM with jumps for capturing crashes and rallies',
    useCases: 'Event risk, fat tails',
  },
  rough_bergomi: {
    name: 'Rough Bergomi',
    description: 'Rough volatility with realistic forward variance curve',
    useCases: 'Advanced vol modeling, research',
  },
};
