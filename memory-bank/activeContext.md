# Active Context

## Current Focus
Planning the Data Pipeline implementation.

## Recent Changes
-   **Version Control**: Initialized Git repository and added `.gitignore`.
-   **Project Structure**: Created `memory-bank` and populated core files.

## Next Steps
1.  **Approving Data Pipeline Plan**: Review the `implementation_plan.md` for fetching BTC data.
2.  **Implementation**: Write the `data_loader.py` script using `ccxt`.
3.  **Environment Setup**: Install necessary dependencies (`ccxt`, `pandas`).

## Active Decisions
-   **Data Source**: Using `ccxt` to fetch from Binance (public API) for simplicity and reliability.
-   **Storage**: Saving data as Parquet (or CSV) for efficient localized loading.
