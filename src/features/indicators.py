import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame using pandas-ta.
    Indicators: RSI, MACD, ATR, Bollinger Bands, ADX, SMA/EMA Crosses.
    
    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume'.
        
    Returns:
        pd.DataFrame: DataFrame with added indicators.
    """
    # Ensure standard column names
    df.columns = [c.lower() for c in df.columns]
    
    # RSI (14)
    # df.ta.rsi(...) returns a Series named RSI_14, we assign it to 'rsi'
    df['rsi'] = df.ta.rsi(length=14)
    
    # MACD (12, 26, 9)
    macd_df = df.ta.macd(fast=12, slow=26, signal=9)
    # Rename MACD columns
    if macd_df is not None:
        # Columns are usually MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        # Helper to find correct columns
        cols = macd_df.columns
        macd_col = [c for c in cols if c.startswith('MACD_')][0]
        hist_col = [c for c in cols if c.startswith('MACDh_')][0]
        signal_col = [c for c in cols if c.startswith('MACDs_')][0]
        
        df['macd'] = macd_df[macd_col]
        df['macd_hist'] = macd_df[hist_col]
        df['macd_signal'] = macd_df[signal_col]
    
    # ATR (14)
    df['atr'] = df.ta.atr(length=14)
    # Avoid division by zero
    df['atr_rel'] = df['atr'] / df['close']
    
    # Bollinger Bands (20, 2)
    bbands = df.ta.bbands(length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        # Use pandas-ta calculated bandwidth (BBB) and percent B (BBP)
        # Find columns dynamically
        bbb_col = [c for c in bbands.columns if c.startswith('BBB')][0]
        bbp_col = [c for c in bbands.columns if c.startswith('BBP')][0]
        
        df['bb_width'] = df[bbb_col]
        df['bb_pos'] = df[bbp_col]

    # ADX (14)
    adx_df = df.ta.adx(length=14)
    if adx_df is not None:
        # ADX usually returns ADX_14, DMP_14, DMN_14
        adx_col = [c for c in adx_df.columns if c.startswith('ADX')][0]
        df['adx'] = adx_df[adx_col]
    
    # SMA/EMA Crosses
    df['sma_20'] = df.ta.sma(length=20)
    df['sma_50'] = df.ta.sma(length=50)
    df['ema_12'] = df.ta.ema(length=12)
    df['ema_26'] = df.ta.ema(length=26)
    
    # Distance from SMAs (normalized)
    df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # Drop rows with NaN (initial warmup period)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
