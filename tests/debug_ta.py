import pandas as pd
import pandas_ta as ta
import numpy as np

dates = pd.date_range(start='2021-01-01', periods=100, freq='1h')
df = pd.DataFrame({
    'timestamp': dates,
    'open': np.random.rand(100) * 100,
    'high': np.random.rand(100) * 100,
    'low': np.random.rand(100) * 100,
    'close': np.random.rand(100) * 100,
    'volume': np.random.rand(100) * 1000
})
df.columns = [c.lower() for c in df.columns]

bbands = df.ta.bbands(length=20, std=2)
print("BBands columns:", bbands.columns.tolist())
