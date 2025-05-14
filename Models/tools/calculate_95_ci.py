import numpy as np
from scipy import stats

def calculate_95_ci(data):
    """
    計算給定數據的95%置信區間。
    
    參數:
    - data (list or np.array): 數據點
    
    返回:
    - tuple: (下限, 上限)
    """
    confidence = 0.95
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # 標準誤
    margin = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return (mean - margin, mean + margin)