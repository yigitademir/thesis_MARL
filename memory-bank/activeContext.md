# Active Context

## Current Focus
Project Initialization and Memory Bank creation. We are establishing the foundational documentation to ensure long-term project coherence for the Master's Thesis.

## Recent Changes
-   Created `memory-bank/` directory.
-   Moved `agents.md` to `memory-bank/agents.md`.
-   Initialized core Memory Bank files (`projectbrief.md`, `productContext.md`, `systemPatterns.md`, `techContext.md`).

## Next Steps
1.  **Data Fetching**: Implement a module to fetch historical BTC data for 5m, 15m, 1h, and 4h timeframes.
2.  **Environment Setup**: Create the Gym-compatible trading environment.
3.  **Agent Baseline**: Train a simple PPO agent on one timeframe to verify the pipeline.

## Active Decisions
-   **Architecture**: Starting with independent agents per timeframe. Coordination will be added later.
-   **Tech**: Using Stable Baselines3 for reliability.
