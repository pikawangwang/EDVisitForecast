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



def bootstrap_calculate_95_ci(data, num_bootstrap=1000):
    """
    使用簡單 bootstrap 方法計算給定數據的 95% 置信區間。
    
    參數:
    - data (list or np.array): 數據點
    - num_bootstrap (int): bootstrap 抽樣次數，預設為 1000
    """
    # 將數據轉為 NumPy 陣列（如果尚未轉換）
    data = np.array(data)
    
    # 儲存 bootstrap 的均值
    bootstrap_means = []
    
    # 進行 num_bootstrap 次抽樣
    for _ in range(num_bootstrap):
        # 隨機放回抽取，產生與原數據相同大小的樣本
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # 計算樣本的均值
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # 計算 95% 置信區間的上下限
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    
    return (lower_bound, upper_bound)