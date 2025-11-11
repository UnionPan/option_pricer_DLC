import yfinance as yf
import pandas as pd
from datetime import datetime


def option_chain_puller(ticker_symbol, output_file=None):
    """
    Pull option chain data for a given ticker symbol and return/save as CSV.

    Parameters:
    -----------
    ticker_symbol : str
        The stock ticker symbol (e.g., 'AAPL', 'TSLA')
    output_file : str, optional
        Path to save the CSV file. If None, returns the dataframe without saving.

    Returns:
    --------
    pd.DataFrame
        Combined dataframe containing both calls and puts option chain data
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Get available expiration dates
        expirations = ticker.options

        if not expirations:
            raise ValueError(f"No option data available for {ticker_symbol}")

        # Get option chain for the first available expiration date
        # You can modify this to get multiple expiration dates
        expiration_date = expirations[0]

        # Get option chain
        opt_chain = ticker.option_chain(expiration_date)

        # Extract calls and puts
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()

        # Add columns to distinguish between calls and puts
        calls['optionType'] = 'call'
        puts['optionType'] = 'put'

        # Add ticker symbol and expiration date
        calls['ticker'] = ticker_symbol
        puts['ticker'] = ticker_symbol
        calls['expirationDate'] = expiration_date
        puts['expirationDate'] = expiration_date

        # Combine calls and puts
        options_df = pd.concat([calls, puts], ignore_index=True)

        # Reorder columns to put identifying info first
        cols = ['ticker', 'expirationDate', 'optionType'] + [col for col in options_df.columns if col not in ['ticker', 'expirationDate', 'optionType']]
        options_df = options_df[cols]

        # Save to CSV if output file is specified
        if output_file:
            options_df.to_csv(output_file, index=False)
            print(f"Option chain data saved to {output_file}")

        return options_df

    except Exception as e:
        print(f"Error pulling option chain data: {e}")
        raise


# Example usage
if __name__ == "__main__":
    # Pull option chain for Apple (AAPL) and save to CSV
    ticker = "AAPL"
    output_csv = f"{ticker}_option_chain_{datetime.now().strftime('%Y%m%d')}.csv"

    df = option_chain_puller(ticker, output_csv)

    print(f"\nOption Chain Summary for {ticker}:")
    print(f"Total options: {len(df)}")
    print(f"Calls: {len(df[df['optionType'] == 'call'])}")
    print(f"Puts: {len(df[df['optionType'] == 'put'])}")
    print(f"\nFirst few rows:")
    print(df.head(10))
