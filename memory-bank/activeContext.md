# Active Context

## Current Focus
Initial Training of Multi-Timeframe Agents.

## Recent Changes
-   **Orchestration**: Implemented `src/launch_training.py` with sequential/parallel/timeframe options.
-   **Verification**: Verified orchestration logic with test runs on all timeframes.

## Next Steps
1.  **Launch Full Training**: Train all agents for 100k+ timesteps (Action: User to run).
2.  **Evaluation**: Implement `src/evaluate_agents.py` to backtest trained models.
3.  **Refinement**: Move to parquet data (Task 16).

## Active Decisions
-   **Workflow**: Sequential training chosen as default to avoid memory overload.
