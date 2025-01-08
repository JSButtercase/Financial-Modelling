# 3a
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_excel("Exam.xlsx")
df = df.iloc[1:]
df = df.rename(columns={"SX5E Index": "Date", "Unnamed: 1": "Price"})
df['daily_returns'] = (df['Price'] - df['Price'].shift())
print(df, "\n", df['daily_returns'].mean())

# 3b
new_df = df.copy(deep=True)
new_df = new_df.drop(columns='daily_returns')
new_df['MA20'] = new_df.rolling(window=20)['Price'].mean()
new_df['MA50'] = new_df.rolling(window=50)['Price'].mean()
new_df = new_df.dropna()
print(new_df)

# 3c
predictors = new_df[["MA20", "MA50"]]
target = new_df["Price"]

training_data_x, testing_data_x, training_data_y, testing_data_y = train_test_split(predictors, target,
                                                                                    test_size=0.2, random_state=32)
reg = LinearRegression()
reg.fit(training_data_x, training_data_y)
print(f"regression coefficient {reg.coef_}")
print(f"regression intercept {reg.intercept_}")

# 3d
# R^2 performance given by confidence from reg.score
confidence = reg.score(testing_data_x, testing_data_y)
print(f'confidence level as a percentage is: {(confidence*100).round(3)}%')
