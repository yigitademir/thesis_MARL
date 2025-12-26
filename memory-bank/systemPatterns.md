# System Patterns

## Architecture Integration
The system follows a standard Deep Reinforcement Learning pipeline with a focus on modularity to support multi-agent experiments.

```mermaid
graph TD
    Data[Data Fetcher] --> Processing[Preprocessing & Feature Eng]
    Processing --> Env{Trading Environment}
    Env --> Agent5m[Agent 5m]
    Env --> Agent15m[Agent 15m]
    Env --> Agent1h[Agent 1h]
    Env --> Agent4h[Agent 4h]
    Agent5m --> Coord[Coordination Layer]
    Agent15m --> Coord
    Agent1h --> Coord
    Agent4h --> Coord
    Coord --> Action[Execution]
```

## Key Technical Decisions
1.  **Stable Baselines3**: Chosen for reliable, bug-free implementations of PPO.
2.  **Per-Timeframe Agents**: Decoupling agents allows for parallel training and specialized feature engineering per timeframe.
3.  **Gymnasium Interface**: Standard environment interface ensures compatibility with SB3 and other RL libraries.

## Design Patterns
-   **Factory Pattern**: For creating environment instances with different configurations (timeframes).
-   **Wrapper Pattern**: Using Gym Wrappers for normalization (`VecNormalize`), stacking, and custom observation transformations.
-   **Strategy Pattern**: For swappable reward functions and action schemes.
