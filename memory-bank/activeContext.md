# Active Context

## Current Focus
Validating the PPO Agent and Trading Environment implementation.

## Recent Changes
-   **Training Loop**: Implemented `src/train_agent.py` with 70/15/15 split.
-   **Verification**: Successfully ran training loop on 4h data (1000 steps).
-   **Dependencies**: Added `tensorboard`, `rich` for visualization.

## Next Steps
1.  **Multi-Agent Training**: Train all 4 agents (5m, 15m, 1h, 4h).
2.  **Evaluation**: Implement `src/evaluate_agents.py` to test performance on Hold-Out data.
3.  **Coordination**: Design the Manager Agent.

## Active Decisions
-   **Data Split**: 70% Train, 15% Val, 15% Test (Chronological).
-   **Optimization**: Deferred Parquet conversion to Phase 2.
