import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.trading_env import TradingEnv
from src.features.indicators import add_indicators

def calculate_metrics(df_history, initial_balance=10000.0):
    """
    Calculate financial metrics from historyDataFrame.
    History cols: [step, portfolio_value, position, price, reward]
    """
    if df_history.empty:
        return {}
    
    # Portfolio Values
    values = df_history['portfolio_value'].values
    returns = df_history['portfolio_value'].pct_change().dropna()
    
    # Total Return
    final_value = values[-1]
    total_return = (final_value - initial_balance) / initial_balance * 100
    
    # Sharpe Ratio (Annualized)
    # Assuming hourly data? Need timeframe info. 
    # Let's approximate based on steps. If 1h, 24*252 steps/year.
    # We will pass 'annual_factor' as arg later, defaulting to 252*24 for now.
    annual_factor = 252 * 24 
    if len(returns) > 0 and returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(annual_factor)
    else:
        sharpe = 0.0
        
    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min() * 100
    
    # Trade Stats
    # Detect position changes
    positions = df_history['position'].values
    trades = 0
    wins = 0
    losses = 0
    
    # Simple logic: count every time position changes from something to something else? 
    # Or count round trips?
    # Let's count "Exits" or "Flips".
    # For simplicity, let's count changes.
    # Actually, to get Win Rate, we need closed trade PnL.
    # Our Env doesn't log trade PnL explicitly in info, but we can infer.
    # Easier: Iterate and track entry/exit.
    
    entry_val = 0
    in_trade = False
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        if prev_pos != curr_pos:
            # Trade happened
            trades += 1
            
    # Win rate calculation requires tracking trade results.
    # Since we don't have per-trade logs in history yet, we might need to rely on Portfolio Value changes over "Trade Duration".
    # This is complex to reconstruct perfectly from just position/value trace without trade logs.
    # Implementation Plan update: We will improve TradingEnv later to return Trade Info.
    # For now, we will approximate or skip intricate Win Rate if data missing.
    # BUT user insisted.
    # Let's backtrack: we can deduce "Win" if we exited a position and Portfolio Value increased since entry?
    
    # Improved Logic:
    # Scan for position changes. 
    # If Pos 0 -> 1 (Open Long). Record Entry Value.
    # If Pos 1 -> 0 (Close Long). Record Exit Value. If Exit > Entry -> Win.
    # If Pos 1 -> -1 (Flip). Close Long (Check Win), Open Short.
    
    trade_pnl = []
    
    current_entry_val = 0
    
    for i in range(1, len(df_history)):
        prev_pos = df_history.iloc[i-1]['position']
        curr_pos = df_history.iloc[i]['position']
        curr_val = df_history.iloc[i]['portfolio_value']
        price = df_history.iloc[i]['price']
        
        if prev_pos == 0 and curr_pos != 0:
            # Open
            current_entry_val = df_history.iloc[i-1]['portfolio_value'] # Approx capital allocated? 
            # Actually, Env uses full balance.
            current_entry_val = curr_val # Value at start of step i
            
        elif prev_pos != 0 and curr_pos != prev_pos:
             # Close or Flip
             # PnL = Val_now - Val_at_entry
             # Wait, portfolio value fluctuates every step.
             # Use (Exit Price - Entry Price) for Long?
             # Env logic: Portfolio Value encapsulates everything (fees, pnl).
             # So: Trade PnL = Val_at_exit - Val_at_entry?
             # Yes, roughly.
             
             pnl = curr_val - current_entry_val
             trade_pnl.append(pnl)
             
             if curr_pos != 0:
                 # Flip: New entry
                 current_entry_val = curr_val
                 
    total_trades = len(trade_pnl)
    if total_trades > 0:
        wins = sum(1 for p in trade_pnl if p > 0)
        win_rate = (wins / total_trades) * 100
        avg_pnl = np.mean(trade_pnl)
    else:
        win_rate = 0.0
        avg_pnl = 0.0

    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Avg PnL': avg_pnl
    }

def evaluate_model(model_path, data_path, timeframe, output_dir="results"):
    print(f"Evaluating {timeframe} model from {model_path}...")
    
    # Load Data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = add_indicators(df)
    
    # Split Test Set (Last 15%)
    n = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    print(f"Test Set Size: {len(test_df)}")
    
    # Load Model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
         print("Model not found.")
         return
         
    model = PPO.load(model_path)
    
    # Run Environment
    env = TradingEnv(test_df)
    obs, info = env.reset()
    
    history = []
    
    # Record Initial
    history.append({
        'step': 0,
        'portfolio_value': info['portfolio_value'],
        'position': info['position'],
        'price': info['price'],
        'reward': 0
    })
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        history.append({
            'step': env.current_step,
            'portfolio_value': info['portfolio_value'],
            'position': info['position'],
            'price': info['price'],
            'reward': reward
        })
        
    df_history = pd.DataFrame(history)
    
    # Metrics
    metrics = calculate_metrics(df_history)
    
    print("\n--- Performance Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    # Plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df_history['portfolio_value'], label='Agent Equity')
    
    # Benchmark (Buy & Hold)
    # Norm to initial balance
    initial_price = df_history['price'].iloc[0]
    initial_bal = df_history['portfolio_value'].iloc[0]
    
    benchmark = (df_history['price'] / initial_price) * initial_bal
    plt.plot(benchmark, label='Buy & Hold (BTC)', alpha=0.6)
    
    plt.title(f"Performance: {timeframe} Agent")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, f"equity_curve_{timeframe}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    
    # Default paths
    clean_tf = args.timeframe
    symbol = "BTC_USDT"
    
    if args.model_path is None:
        # Try to find best model first, then last model
        best_path = f"models/{symbol}_{clean_tf}/best_model.zip"
        last_path = f"models/{symbol}_{clean_tf}/last_model.zip"
        
        if os.path.exists(best_path):
            args.model_path = best_path
        elif os.path.exists(last_path):
            args.model_path = last_path
        else:
            # Just try last_model default
            args.model_path = last_path
            
    data_path = f"data/{symbol}_{clean_tf}.csv"
    
    evaluate_model(args.model_path, data_path, clean_tf)

if __name__ == "__main__":
    main()
