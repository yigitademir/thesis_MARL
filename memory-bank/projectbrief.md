# Project Brief

## Overview
Development of a Multi-Agent Proximal Policy Optimization (PPO) Framework for Cryptocurrency Trading, specifically targeting Bitcoin (BTC) across multiple timeframes (5m, 15m, 1h, 4h). The project aims to train expert agents for each timeframe and subsequently evaluate coordination mechanisms for a Master's Thesis.

## Core Requirements
1.  **Multi-Agent System**: Distinct agents for 5m, 15m, 1h, and 4h timeframes.
2.  **Algorithm**: Proximal Policy Optimization (PPO) using Stable Baselines3.
3.  **Asset**: Bitcoin (BTC).
4.  **Data Pipeline**: Robust data fetching and preprocessing system.
5.  **Coordination**: Mechanisms to coordinate agent decisions (Ensemble/Manager) - *Phase 2*.
6.  **Environment**: Custom trading environment compatible with Gym/Gymnasium.

## Goals
-   Train individual PPO agents to become experts in their respective timeframes.
-   Implement a data fetching module for historical crypto data.
-   Develop and evaluate coordination strategies between agents.
-   Produce a rigorous codebase suitable for a Master's Thesis.
