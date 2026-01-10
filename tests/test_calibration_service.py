"""
Test calibration service for both P-measure and Q-measure
"""
import sys
from pathlib import Path

# Add backend and src to path
backend_path = str(Path(__file__).parent.parent / "backend")
src_path = str(Path(__file__).parent.parent / "src")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from backend.services.calibration_service import CalibrationService
from datetime import datetime, date, timedelta


def test_pmeasure_fetch_ohlcv():
    """Test fetching OHLCV data for P-measure calibration"""
    print("\n=== Testing P-measure: Fetch OHLCV ===")

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    try:
        data = CalibrationService.fetch_ohlcv_data(
            tickers=['SPY'],
            start_date=start_date,
            end_date=end_date,
        )

        print(f"✓ Successfully fetched OHLCV data")
        print(f"  Ticker: {data[0]['ticker']}")
        print(f"  Data points: {len(data[0]['dates'])}")
        print(f"  First date: {data[0]['dates'][0]}")
        print(f"  Last date: {data[0]['dates'][-1]}")
        print(f"  Last close: ${data[0]['close'][-1]:.2f}")

        return data

    except Exception as e:
        print(f"✗ Failed to fetch OHLCV: {e}")
        return None


def test_pmeasure_calibrate_gbm():
    """Test P-measure GBM calibration"""
    print("\n=== Testing P-measure: Calibrate GBM ===")

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    try:
        # Fetch data first
        data = CalibrationService.fetch_ohlcv_data(
            tickers=['SPY'],
            start_date=start_date,
            end_date=end_date,
        )

        # Calibrate
        import numpy as np
        prices = np.array(data[0]['close'])
        dates = data[0]['dates']

        parameters, diagnostics = CalibrationService.calibrate_model(
            ticker='SPY',
            prices=prices,
            dates=dates,
            model='gbm',
            method='mle',
            include_drift=True,
        )

        print(f"✓ Successfully calibrated GBM model")
        print(f"  Parameters:")
        for key, value in parameters.items():
            print(f"    {key}: {value:.6f}")
        print(f"  Diagnostics:")
        for key, value in diagnostics.items():
            if value is not None and not isinstance(value, dict):
                print(f"    {key}: {value}")

        return parameters, diagnostics

    except Exception as e:
        print(f"✗ Failed to calibrate GBM: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_qmeasure_fetch_option_chain():
    """Test fetching option chain for Q-measure calibration"""
    print("\n=== Testing Q-measure: Fetch Option Chain ===")

    try:
        chain_data = CalibrationService.fetch_option_chain(
            ticker='SPY',
            reference_date=None,  # Use today
            risk_free_rate=0.05,
            expiry=None,  # Get all expiries
        )

        print(f"✓ Successfully fetched option chain")
        print(f"  Ticker: {chain_data['ticker']}")
        print(f"  Spot price: ${chain_data['spot_price']:.2f}")
        print(f"  Reference date: {chain_data['reference_date']}")
        print(f"  Total options: {chain_data['n_options']}")
        print(f"  Available expiries: {len(chain_data['expiries'])}")
        print(f"  First few expiries: {chain_data['expiries'][:3]}")

        return chain_data

    except Exception as e:
        print(f"✗ Failed to fetch option chain: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_qmeasure_calibrate_heston():
    """Test Q-measure Heston calibration"""
    print("\n=== Testing Q-measure: Calibrate Heston ===")

    try:
        parameters, diagnostics = CalibrationService.calibrate_qmeasure_model(
            ticker='SPY',
            model='heston',
            reference_date=None,
            risk_free_rate=0.05,
            filter_params={
                'min_volume': 10,
                'min_open_interest': 50,
                'max_spread_pct': 0.5,
                'moneyness_range': [0.8, 1.2],
            },
            calibration_method='differential_evolution',
            maxiter=100,  # Use fewer iterations for testing
        )

        print(f"✓ Successfully calibrated Heston model")
        print(f"  Parameters:")
        for key, value in parameters.items():
            print(f"    {key}: {value:.6f}")
        print(f"  Diagnostics:")
        for key, value in diagnostics.items():
            if value is not None and not isinstance(value, dict):
                print(f"    {key}: {value}")
        print(f"  Feller condition: {diagnostics.get('feller_condition', 'N/A')}")

        return parameters, diagnostics

    except Exception as e:
        print(f"✗ Failed to calibrate Heston: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Run all tests"""
    print("=" * 60)
    print("Calibration Service Test Suite")
    print("=" * 60)

    # Test P-measure
    print("\n" + "=" * 60)
    print("P-MEASURE CALIBRATION TESTS")
    print("=" * 60)
    test_pmeasure_fetch_ohlcv()
    test_pmeasure_calibrate_gbm()

    # Test Q-measure
    print("\n" + "=" * 60)
    print("Q-MEASURE CALIBRATION TESTS")
    print("=" * 60)
    test_qmeasure_fetch_option_chain()

    print("\n" + "=" * 60)
    print("Q-MEASURE CALIBRATION (This may take a minute...)")
    print("=" * 60)
    test_qmeasure_calibrate_heston()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
