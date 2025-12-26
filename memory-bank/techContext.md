# Tech Context

## Technology Stack
-   **Language**: Python 3.9+
-   **Deep Learning**: PyTorch
-   **RL Library**: Stable Baselines3 (+ SB3 Contrib if needed)
-   **Environment**: Gymnasium (formerly OpenAI Gym)
-   **Data Analysis**: Pandas, NumPy
-   **Technical Analysis**: TA-Lib / Pandas-TA
-   **Data Fetching**: CCXT (implied for crypto) or yfinance

## Development Environment
-   **OS**: Mac (User's system)
-   **Dependency Management**: `pip` / `requirements.txt` or `conda` environment.
-   **Version Control**: Git

## Constraints
-   **Performance**: Training multiple PPO agents can be computationally intensive. Efficient data loading and vectorization (`VecEnv`) are critical.
-   **Data Quality**: Crypto data can be noisy. Robust cleaning and preprocessing are required.
