# Product Context

## Purpose
 This project serves as the practical implementation for a Master's Thesis investigating the efficacy of Multi-Agent Reinforcement Learning (MARL) in cryptocurrency markets. It addresses the challenge of non-stationarity and multi-scale patterns in financial time series by decomposing the problem across different time horizons.

## Problem Statement
Cryptocurrency markets are highly volatile and exhibit distinct patterns across different timeframes. A single-agent approach often struggles to capture both short-term noise and long-term trends simultaneously.

## Proposed Solution
A Multi-Agent framework where:
-   **Specialized Agents**: Each agent focuses on a specific timeframe (5m, 15m, 1h, 4h), allowing them to learn timeframe-specific alpha.
-   **Coordination**: A higher-level mechanism (to be defined) synthesizes these diverse "opinions" into a cohesive trading strategy, potentially improving robustness and risk-adjusted returns compared to single-agent baselines.

## User Experience Goals
-   **Reproducibility**: Clear seed management and configuration for scientific validity.
-   **Modularity**: Easy to swap agents, timeframes, or assets.
-   **Observability**: Clear logs and metrics (via Tensorboard/WandB) to monitor training progress and agent behaviors.
