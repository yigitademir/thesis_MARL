# Active Context

## Current Focus
Evaluating trained Multi-Timeframe Agents.

## Recent Changes
-   **Optimization**: Converted data to Parquet format for faster loading.
-   **Training**: Completed full 100k step training for all agents (5m, 15m, 1h, 4h).
-   **Evaluation**: Ready to run `evaluate_agents.py` on the trained models.

## Next Steps
1.  **Run Evaluation**: Generate metrics and equity curves for all 4 agents.
2.  **Analysis**: Review the results. If agents are profitable/promising, proceed to Coordination. If not, debug strategies.
3.  **Coordination**: Design the Manager Agent.

## Active Decisions
-   **Storage**: Parquet adopted for performance.
