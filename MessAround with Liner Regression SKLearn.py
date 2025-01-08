import numpy as np
from matplotlib import pyplot as plt
import warnings
import sklearn.linear_model as sk

np.random.seed(10)
warnings.filterwarnings("ignore")

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

x_b = np.c_[np.ones((100, 1)), x]
x_b_T = x_b.T

# βhat=(X^T * X)^−1 * (X^T * y)

β_hat = np.linalg.inv(x_b_T.dot(x_b)).dot(x_b_T).dot(y)
print(f"β_hat =\n{β_hat}\n")

new_x = np.array([[0], [2]])
new_x_b = np.c_[np.ones((2, 1)), new_x]

model_y = new_x_b.dot(β_hat)
print(f"Predict range =\n{model_y}\n")

plt.plot(new_x, model_y, "r-", linewidth=2, label="Predictions")
plt.plot(x, y, "b.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, 0, 15])
plt.show()

lin = sk.LinearRegression()
lin.fit(x, y)
print("β_hat from sklearn = \n", lin.intercept_, lin.coef_, "\n")
print("prediction from sklearn = \n", lin.predict(new_x), "\n")

def gradient_desc(start, learn_rate, dataset_len, max_iter):
    beta = start
    steps = []

    for i in range(max_iter):
        gradient = 2/dataset_len * x_b_T.dot(x_b.dot(beta)-y)
        beta = beta - learn_rate * gradient
        steps.append(beta)

    return beta, steps


example, history = gradient_desc(np.random.randn(2, 1), 0.02, 100, 100)
ex_predict = new_x_b.dot(example)

print(f"starting at Beta, gradient desc returns \n{ex_predict}")


def plotting():
    plt.plot(x, y, "b.")
    for i in range(len(history)):
        if i < 10:
            model_y = new_x_b.dot(np.array(history[i]))
            style = "r--" if i < 1 else "b-" if i < 9 else 'g-'  # red dashed for 1st, then blue solid, green for last
            plt.plot(new_x, model_y, style)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 14])

plotting()
plt.show()