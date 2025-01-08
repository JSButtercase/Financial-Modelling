# a1
class Stock:
    index = 'SPX'

    def __init__(self, ticker, price):
        self.ticker = ticker
        self.price = price


Apple = Stock('AAPL', [100, 20, 60, 110])
#print(Apple.__dir__())

# a2
class StockPort(Stock):
    index = 'CAC40'


Microsoft = StockPort('MSFT', [50])
#print(Microsoft.index)

# a3
class Trader():
    def __init__(self, price, action=[], number=0):
        assert isinstance(action, list) & len(action) == 0
        assert number == 0
        self.price = price
        self.action = action
        self.number = number


    def takeAction(self):
        for p in self.price:
            if p < 50:
                self.action.append('Buy')
                self.number = self.number + 1
            elif p > 90:
                self.action.append('Sell')
                self.number = self.number - 1
            else:
                self.action.append('Hold')


test_trader = Trader(Apple.price)
test_trader.takeAction()
print(f'Apple test trader actions taken = {test_trader.action}')
print(f'Apple test trader number = {test_trader.number}')

# b1
import numpy as np
import scipy as sp
# dot product of B and B transpose gives A for positive semidefinite
def matrix_generator(n):
    B = np.random.randn(n, n)
    B_T = np.transpose(B)
    return np.dot(B, B_T)


#b2
A = matrix_generator(4)
print(A)

# use scipy.linalg.eigh to evaluate eigenvalues as non negative for A:

eigs = sp.linalg.eigh(A, eigvals_only=True)
for element in eigs:
    if element < 0:
        print("A is not positive semi-definite")
        break
    else:
        print("A is positive semi-definite")

print(eigs)



