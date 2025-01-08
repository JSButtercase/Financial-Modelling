import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


exam_excel = pd.read_excel('ExamExcel.xlsx')
print(type(exam_excel), "\n")                    # exam_excel is a dataframe
columns = exam_excel.columns.tolist()            # gives the columns of the dataset provided as a list
print(columns, "\n")

# drops the duplicated dates.x columns
exam_excel = exam_excel.drop(exam_excel.columns[[2, 4, 6, 8]], axis=1)
columns_fixed = exam_excel.columns.tolist()       # gives the updated columns  as a list
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

print(new_df.head(5))   # ta daa!

log_df = pd.DataFrame(columns=['lgAC', 'lgAI', 'lgALO', 'lgBN', 'lgBNP'])
daily_returns = pd.DataFrame(columns=['dailyAC', 'dailyAI', 'dailyALO', 'dailyBN', 'dailyBNP'])
std = pd.DataFrame(columns=['stdAC', 'stdAI', 'stdALO', 'stdBN', 'stdBNP'])

for i in range(len(log_df.columns.tolist())):
    j = new_df.columns[i+1]
    k = log_df.columns[i]
    l = daily_returns.columns[i]

    log_df[k] = np.log(new_df[j]/new_df[j].shift(1))
    daily_returns[l] = new_df[j].pct_change(1) * 100

rolling_window = 250

for i in range(len(std.columns.tolist())):
    m = daily_returns.columns[i]
    n = std.columns[i]
    std[n] = daily_returns[m].rolling(rolling_window).std()
    plt.plot(daily_returns[m], label=m)
    plt.legend(loc='lower left')
    plt.title(m)
    plt.show()

std = std.dropna()
log_df = log_df.dropna()
daily_returns = daily_returns.dropna()
print(f"log returns = \n{log_df}\n expected daily returns = \n{daily_returns} \n")
print(f"standard deviations aka volatility in a rolling 250 day window = \n{std}")

#Var0


