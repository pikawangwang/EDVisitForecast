import pandas as pd
import numpy as np
import xarray as xr
from darts import TimeSeries
from scipy import stats
import matplotlib.pyplot as plt
from darts.metrics.metrics import mape, mae, mse, ape

def calculate_metrics(all_series, yearly_dataarrays, APEdf, calculate_95_ci):
    """
    計算各種預測指標，包括 MAE、MAPE 和 APE 的 95% 置信區間。

    參數:
    - all_series (TimeSeries): 完整的實際時間序列數據。
    - yearly_dataarrays (dict): 字典，鍵為年份，值為預測的 xarray DataArray。
    - APEdf (pd.DataFrame): 包含 APE 值的 DataFrame，按日期索引。
    - calculate_95_ci (function): 計算 95% 置信區間的函數。

    返回:
    - dict: 包含所有計算結果的字典。
    """
    
    metrics = {}
    ape_2018_2019 = []
    ape_2020_2021 = []
    ape_2022 = []

    # 初始化列表以存儲實際值和預測值，便於後續計算綜合 MAE
    actual_2018_2019 = []
    forecast_2018_2019 = []
    actual_2020_2021 = []
    forecast_2020_2021 = []
    actual_2022 = []
    forecast_2022 = []

    for year in range(2018, 2023):
        actual = all_series.slice(pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31'))
        forecast = TimeSeries.from_xarray(yearly_dataarrays[year])
        
        mae_value = mae(actual, forecast)
        mape_value = mape(actual, forecast)
        
        ape_yearly = APEdf.loc[f'{year}-01-01':f'{year}-12-31', 'APE']
        
        if year in [2018, 2019]:
            ape_2018_2019.extend(ape_yearly)
            actual_2018_2019.append(actual.pd_dataframe())
            forecast_2018_2019.append(forecast.pd_dataframe())
        elif year in [2020, 2021]:
            ape_2020_2021.extend(ape_yearly)
            actual_2020_2021.append(actual.pd_dataframe())
            forecast_2020_2021.append(forecast.pd_dataframe())
        elif year == 2022:
            ape_2022.extend(ape_yearly)
            actual_2022.append(actual.pd_dataframe())
            forecast_2022.append(forecast.pd_dataframe())
        
        ci_95 = calculate_95_ci(ape_yearly)
        
        metrics[year] = {
            "MAE": mae_value,
            "MAPE": mape_value,
            "APE 95% CI": ci_95
        }
    
    # Combine and calculate metrics for periods
    actual_2018_2019_combined = pd.concat(actual_2018_2019)
    forecast_2018_2019_combined = pd.concat(forecast_2018_2019)
    actual_2020_2021_combined = pd.concat(actual_2020_2021)
    forecast_2020_2021_combined = pd.concat(forecast_2020_2021)
    actual_2022_combined = pd.concat(actual_2022)
    forecast_2022_combined = pd.concat(forecast_2022)
    
    # Calculate combined MAE and confidence intervals
    mae_2018_2019_combined = mae(TimeSeries.from_dataframe(actual_2018_2019_combined), 
                                 TimeSeries.from_dataframe(forecast_2018_2019_combined))
    mae_2020_2021_combined = mae(TimeSeries.from_dataframe(actual_2020_2021_combined), 
                                 TimeSeries.from_dataframe(forecast_2020_2021_combined))
    mae_2022_combined = mae(TimeSeries.from_dataframe(actual_2022_combined), 
                            TimeSeries.from_dataframe(forecast_2022_combined))
    
    ci_95_2018_2019 = calculate_95_ci(ape_2018_2019)
    ci_95_2020_2021 = calculate_95_ci(ape_2020_2021)
    ci_95_2022 = calculate_95_ci(ape_2022)
    
    # Combine data for all years
    actual_all = pd.concat([actual_2018_2019_combined, actual_2020_2021_combined, actual_2022_combined])
    forecast_all = pd.concat([forecast_2018_2019_combined, forecast_2020_2021_combined, forecast_2022_combined])
    mae_all = mae(TimeSeries.from_dataframe(actual_all), TimeSeries.from_dataframe(forecast_all))
    
    # Calculate metrics for combined periods
    combined_metrics = {
        "2018-2019 Combined": {
            "MAE": mae_2018_2019_combined,
            "MAPE": np.mean(ape_2018_2019),
            "APE 95% CI": ci_95_2018_2019
        },
        "2020-2021 Combined": {
            "MAE": mae_2020_2021_combined,
            "MAPE": np.mean(ape_2020_2021),
            "APE 95% CI": ci_95_2020_2021
        },
        "2022": {
            "MAE": mae_2022_combined,
            "MAPE": np.mean(ape_2022),
            "APE 95% CI": ci_95_2022
        },
        "2018-2022 Combined": {
            "MAE": mae_all,
            "MAPE": np.mean(APEdf['APE']),
            "APE 95% CI": calculate_95_ci(APEdf['APE'])
        }
    }
    
    metrics.update(combined_metrics)
    
    return metrics  # Now return the metrics instead of printing them
