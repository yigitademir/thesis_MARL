
import sys
import os
sys.path.append(os.getcwd())

from src.agents.ensembles import AssetWeightedEnsemble, VotingEnsemble
from unittest.mock import MagicMock
import numpy as np

def test_ensembles():
    print("\n--- Testing Ensembles ---")
    
    # Mock Models
    m1 = MagicMock()
    m1.predict.return_value = (1, None) # Long
    m2 = MagicMock()
    m2.predict.return_value = (0, None) # Hold
    m3 = MagicMock()
    m3.predict.return_value = (2, None) # Short
    m4 = MagicMock()
    m4.predict.return_value = (1, None) # Long
    
    models = {'A': m1, 'B': m2, 'C': m3, 'D': m4}
    
    # 1. Voting Ensemble
    ve = VotingEnsemble(models)
    action, _ = ve.predict(None)
    print(f"VotingEnsemble Prediction: {action}") 
    # 1, 0, 2, 1 -> Mode is 1 (Long). 
    assert action == 1
    
    # 2. Asset Weighted Ensemble
    awe = AssetWeightedEnsemble(models, initial_capital=10000)
    
    # Scenario: Agent A has double capital
    awe.portfolios['A'] = 20000
    awe.portfolios['B'] = 10000
    awe.portfolios['C'] = 10000
    awe.portfolios['D'] = 10000
    # Total = 50k. A=0.4, others=0.2
    
    # Votes: A=Long(1), B=Hold(0), C=Short(-1), D=Long(1)
    # Score = 1*0.4 + 0*0.2 + (-1)*0.2 + 1*0.2
    # Score = 0.4 - 0.2 + 0.2 = 0.4
    # Threshold > 0.33 -> Long (1)
    
    action, weights = awe.predict(None)
    print(f"Weighted Prediction: {action} (Score shd be 0.4)")
    print(f"Weights: {weights}")
    assert action == 1
    assert weights['A'] == 0.4
    
    print("Ensemble Tests Passed!")

if __name__ == "__main__":
    test_ensembles()
