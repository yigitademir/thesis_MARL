# Active Context

## Current Focus
Training Sub-Agents (Step 1/2) and Evaluating Coordination (Step 2/2).

## Recent Changes
-   **Coordination**: Implemented `ManagerAgent`, `VotingEnsemble`, `AssetWeightedEnsemble`.
-   **Verification**: Unit tests passed for `CoordinationEnv` and Ensembles.
-   **Training**: Sub-agents (5m, 15m, 1h, 4h) are training (1M steps) in background.

## Next Steps
1.  **Wait for Training**: Allow `launch_training.py` to finish (Estimated: 2-3 hours).
2.  **Train Manager**: Run `src/train_manager.py` to train the PPO Meta-Learner.
3.  **Final Evaluation**: Compare Single Best vs. Voting vs. Weighted vs. Manager.

## Active Decisions
-   **Ensembles**: Added Asset-Weighted Voting as a dynamic baseline per user request.
