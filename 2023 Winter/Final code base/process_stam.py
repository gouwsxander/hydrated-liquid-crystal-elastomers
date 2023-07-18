import numpy as np

RH_10 = np.array([0.020, 0.026, 0.018, 0.013])
RH_40 = np.array([0.040, 0.064, 0.033, 0.031])
RH_60 = np.array([0.055, 0.078, 0.048, 0.051])
RH_90 = np.array([0.104, 0.124, 0.104, 0.094])

rows = [RH_10, RH_40, RH_60, RH_90]

for row in rows:
    print(np.mean(row), np.std(row, ddof=1)/np.sqrt(len(row)))