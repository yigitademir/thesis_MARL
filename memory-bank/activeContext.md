# Active Context

## Current Focus
Training and Evaluating Multi-Timeframe Agents.

## Recent Changes
-   **Evaluation**: Implemented `src/evaluate_agents.py` (Sharpe, MaxDD, Trades).
-   **Environment**: Fixed `reset()` to return info dict for tracking portfolio value.
-   **Training**: Launched full 100k step training for all agents.

## Next Steps
1.  **Analyze Results**: Wait for training to finish and run evaluation on all timeframes.
2.  **Coordination**: Design the Manager Agent to combine these sub-agents.
3.  **Optimization**: Convert CSV to Parquet.

## Active Decisions
-   **Metrics**: Added "Total Trades" and "Win Rate" to evaluation at user request.
