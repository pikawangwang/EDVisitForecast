import pandas as pd
import numpy as np
import xarray as xr
from darts import TimeSeries
from scipy import stats
import matplotlib.pyplot as plt
from darts.metrics.metrics import mape, mae, mse, ape

def calculate_metrics(all_series, yearly_dataarrays, APEdf, AEdf, calculate_95_ci):
    """
    計算各種預測指標，包括 MAE、MAPE 的 95% 置信區間。

    參數:
    - all_series (TimeSeries): 完整的實際時間序列數據。
    - yearly_dataarrays (dict): 字典，鍵為年份，值為預測的 xarray DataArray。
    - APEdf (pd.DataFrame): 包含 APE 值的 DataFrame，按日期索引。
    - AEdf (pd.DataFrame): 包含 AE 值的 DataFrame，按日期索引。
    - calculate_95_ci (function): 計算 95% 置信區間的函數。

    返回:
    - dict: 包含所有計算結果的字典。
    """
    
    metrics = {}
    ape_2018_2019 = []
    ae_2018_2019 = []
    ape_2020_2021 = []
    ae_2020_2021 = []
    ape_2022 = []
    ae_2022 = []

    for year in range(2018, 2023):
        # 提取實際值和預測值的 TimeSeries
        actual = all_series.slice(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
        forecast = TimeSeries.from_xarray(yearly_dataarrays[year])
        
        # 計算年度指標
        mae_value = mae(actual, forecast)
        mape_value = mape(actual, forecast)

        # 提取 APE 和 AE 值
        ape_yearly = APEdf.loc[f'{year}-01-01':f'{year}-12-31', 'APE']
        ae_yearly = AEdf.loc[f'{year}-01-01':f'{year}-12-31', 'AE']

        # 根據年份分類存儲 APE 和 AE
        if year in [2018, 2019]:
            ape_2018_2019.extend(ape_yearly)
            ae_2018_2019.extend(ae_yearly)
        elif year in [2020, 2021]:
            ape_2020_2021.extend(ape_yearly)
            ae_2020_2021.extend(ae_yearly)
        elif year == 2022:
            ape_2022.extend(ape_yearly)
            ae_2022.extend(ae_yearly)
        
        # 計算置信區間
        ci_95_ape = calculate_95_ci(ape_yearly)
        ci_95_ae = calculate_95_ci(ae_yearly)
        
        # 存儲年度指標
        metrics[year] = {
            "MAE": mae_value,
            "MAPE": mape_value,
            "APE 95% CI": ci_95_ape,
            "AE 95% CI": ci_95_ae
        }
    
    # 計算合併期間的指標
    combined_metrics = {
        "2018-2019 Combined": {
            "MAE": np.mean(ae_2018_2019),  # 使用 AE 的平均值計算
            "MAPE": np.mean(ape_2018_2019),
            "APE 95% CI": calculate_95_ci(ape_2018_2019),
            "AE 95% CI": calculate_95_ci(ae_2018_2019)
        },
        "2020-2021 Combined": {
            "MAE": np.mean(ae_2020_2021),  # 使用 AE 的平均值計算
            "MAPE": np.mean(ape_2020_2021),
            "APE 95% CI": calculate_95_ci(ape_2020_2021),
            "AE 95% CI": calculate_95_ci(ae_2020_2021)
        },
        "2022": {
            "MAE": np.mean(ae_2022),  # 使用 AE 的平均值計算
            "MAPE": np.mean(ape_2022),
            "APE 95% CI": calculate_95_ci(ape_2022),
            "AE 95% CI": calculate_95_ci(ae_2022)
        },
        "2018-2022 Combined": {
            "MAE": np.mean(AEdf['AE']),  # 使用 AE 的平均值計算
            "MAPE": np.mean(APEdf['APE']),
            "APE 95% CI": calculate_95_ci(APEdf['APE']),
            "AE 95% CI": calculate_95_ci(AEdf['AE'])
        }
    }
    
    metrics.update(combined_metrics)
    
    return metrics