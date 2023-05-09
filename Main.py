# Load packages
import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

PATHS = {
    'AGES_Christofides': './Data/AGES_Christofides.csv',
    'EAX-MA_Christofides': './Data/EAX-MA_Christofides.csv',
    'AGES_Golden':'./Data/AGES_Golden.csv',
    'EAX-MA_Christofides':'./Data/EAX-MA_Christofides.csv',
    'EAX-MA_Golden':'./Data/EAX-MA_Golden.csv',
    'HGSADC_Christofides':'./Data/HGSADC_Christofides.csv',
    'HGSADC_Golden':'./Data/HGSADC_Golden.csv',
    'ILS-RVND-SP_Christofides':'./Data/ILS-RVND-SP_Christofides.csv',
    'ILS-RVND-SP_Golden':'./Data/ILS-RVND-SP_Golden.csv'
}

def lmem_normality(name):
    data = pd.read_csv(PATHS[name])
    data = pd.melt(data, id_vars=["Instance"], var_name="Method", value_name="Value")

    # Run LMER
    model = smf.mixedlm("Value ~ Method", data=data, groups=data["Instance"])
    result = model.fit()

    log_data = copy.deepcopy(data)
    log_data['Value'] = np.log(log_data['Value'])
    log_model = smf.mixedlm("Value ~ Method", data=log_data, groups=data["Instance"])
    log_result = log_model.fit()
    # Obtain the residuals
    residuals = result.resid

    # Print the residual values for each group
    df_results = pd.DataFrame()
    for group, values in residuals.groupby(data["Instance"]):
        statistic, p_value = stats.shapiro(values)
        df_results.loc[group,'normal']= p_value

    log_residuals = log_result.resid
    for group, values in log_residuals.groupby(log_data["Instance"]):
        statistic, p_value = stats.shapiro(values)
        df_results.loc[group,'log']= p_value


    df_results.to_csv(f'./Data/Results/{name}.csv')

for name in PATHS.keys():
    lmem_normality(name)