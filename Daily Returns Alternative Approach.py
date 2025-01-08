import pandas as pd
import numpy as np
import scipy.stats as scs

pd.set_option('display.max_columns', None)

exam_excel = pd.read_excel('ExamExcel.xlsx')
print(type(exam_excel), "\n")                    # creates a dataframe
columns = exam_excel.columns.tolist()            # gives the names of the columns, provided as a list
print(columns, "\n")

# drops the duplicated dates.x columns
exam_excel = exam_excel.drop(exam_excel.columns[[2, 4, 6, 8]], axis=1)
columns_fixed = exam_excel.columns.tolist()       # gives the updated columns as a list
print(columns_fixed, "\n", exam_excel)

# pandas has a great inbuilt tool for linear interpolation that works a treat here
# first replace the 0's with NA so that pandas 'drop' works
exam_excel.replace(0, np.nan, inplace=True)
# date formats dont mix well with interpolate, and as we dont need it, we can drop it for now
new_df = exam_excel.drop('Date', axis=1)
# interpolate by having limit_area inside, and the described task is linear, so thats what we set method to
new_df = new_df.interpolate(method='linear', limit_area='inside', limit_direction='forward')
# now we can add the date field back
new_df.insert(0, 'Date', exam_excel['Date'])

print(new_df.head(5), "\n")   # ta daa!

# create some new dataframes for parts (c) and (d)
log_df = pd.DataFrame(columns=['lgAC', 'lgAI', 'lgALO', 'lgBN', 'lgBNP'])
daily_returns = pd.DataFrame(columns=['dailyAC', 'dailyAI', 'dailyALO', 'dailyBN', 'dailyBNP'])
std = pd.DataFrame(columns=['stdAC', 'stdAI', 'stdALO', 'stdBN', 'stdBNP'])

rolling_window = 250
"""
loops over as many columns as there are in the logreturns dataframe
uses the iterator to get the names of columns 
each iteration calcs the logret of that col & the daily return of that col too
"""
for i in range(len(log_df.columns.tolist())):
    j = new_df.columns[i+1]
    k = log_df.columns[i]
    log_df[k] = np.log(new_df[j]/new_df[j].shift(1))
    new_df['daily_returns_'+j] = (new_df[j].pct_change(1)) * 100

new_df = new_df.dropna()

"""
similar to the last loop, but for calculating the rolling average mean daily return
and the rolling standard deviation over the same period
"""
for i in range(len(daily_returns.columns.tolist())):
    l = std.columns[i]
    m = new_df.columns[i+6]
    n = daily_returns.columns[i]
    daily_returns[n] = new_df[m].rolling(rolling_window).mean()
    std[l] = new_df[m].rolling(rolling_window).std()

std = std.dropna()
log_df = log_df.dropna()
daily_returns = daily_returns.dropna()

print(f"log returns = \n{log_df}\n mean daily returns = \n{daily_returns} \n")
print(f"standard deviations of daily returns = \n{std}")

""" 
========
VaR calc
========
mu = daily returns (mean value)
sigma = std  (standard deviation of expected returns)
alpha = 0.01 (1%)
VaR = -scs.norm.ppf(alpha, loc=mu, scale=sigma)
"""

# same as before, a dataframe to store the rolling parametric normal VaR for each stock
VaR = pd.DataFrame(columns=['VaR_AC', 'VaR_AI', 'VaR_ALO', 'VaR_BN', 'VaR_BNP'])

for i in range(len(daily_returns.columns.tolist())):
    _ = VaR.columns[i]
    mu = daily_returns[daily_returns.columns[i]]
    sigma = std[std.columns[i]]
    alpha = 0.01
    VaR[_] = -scs.norm.ppf(alpha, loc=mu, scale=sigma)

# VaR using moving average mean/std calculated in previous step
print(f"VaR of each stocks daily returns \n{VaR}\n")
# Get the last value for each stock
for i in range(len(VaR.columns.tolist())):
    print(f"last estimate for {VaR.columns[i]}: {VaR[VaR.columns[i]].iloc[VaR.shape[0]-1]}")
