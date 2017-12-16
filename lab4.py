import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
boston.data.shape

print boston.feature_names
print boston.DESCR

bos = pd.DataFrame(boston.data)
bos.head()

bos.columns = boston.feature_names
bos.head()

print boston.target.shape
bos['PRICE'] = boston.target
bos.head()

bos.describe()

plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")
plt.show()

plt.scatter(bos.RM, bos.PRICE, alpha=0.4)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
plt.show()

sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)
plt.show()

plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")
plt.show()

plt.hist(bos.CRIM)
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequencey")
plt.show()

plt.hist(bos.PRICE, bins=30)
plt.title('Housing Prices: $Y_i$')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

import statsmodels.api as sm
from statsmodels.formula.api import ols

m = ols('PRICE ~ RM', bos).fit()
print m.summary()

plt.scatter(bos['PRICE'],m.fittedvalues)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$")

from sklearn.linear_model import LinearRegression
X = bos.drop("PRICE", axis = 1)

lm = LinearRegression()
lm

lm.fit(X, bos.PRICE)
lm2 = LinearRegression(fit_intercept=False)
lm2.fit(X, bos.PRICE)

print 'Estimated intercept coefficient:', lm.intercept_
print 'Estimated intercept coefficient:', lm2.intercept_

print 'Number of coefficients:', len(lm.coef_)
pd.DataFrame(zip(X.columns, lm.coef_), columns=["features", "estimatedCoefficients"])

lm.predict(X)[0:5]

plt.hist(lm.predict(X))
plt.title('Predicted Housing Prices (fitted values): $\hat{Y}_i$')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.scatter(bos.PRICE, lm.predict(X))
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$")

print np.sum((bos.PRICE - lm.predict(X)) ** 2)

mseFull = np.mean((bos.PRICE - lm.predict(X)) ** 2)
print mseFull

lm = LinearRegression()
lm.fit(X[["PTRATIO"]], bos.PRICE)

msePTRATIO = np.mean((bos.PRICE - lm.predict(X[["PTRATIO"]])) ** 2)
print msePTRATIO

plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")
plt.plot(bos.PTRATIO, lm.predict(X[['PTRATIO']]), color='red', linewidth=3)
plt.show()

lm = LinearRegression()
lm.fit(X[["CRIM","RM","PTRATIO"]], bos.PRICE)
mseCrimRmPtratio = np.mean((bos.PRICE - lm.predict(X[["CRIM","RM","PTRATIO"]])) ** 2)
print mseCrimRmPtratio

X_train = X[:-50]
X_test = X[-50:]
Y_train = bos.PRICE[:-50]
Y_test = bos.PRICE[-50:]
print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

import sklearn.cross_validation
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, bos.PRICE, test_size=0.33, random_state = 5)

lm = LinearRegression()
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

print "Fit a model X_train, and calculate MSE with Y_train:", np.mean((Y_train - lm.predict(X_train)) ** 2)
print "Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((Y_test - lm.predict(X_test)) ** 2)

plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c='b', s=40, alpha=0.5)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')

faithful = sm.datasets.get_rdataset("faithful")
sm.datasets.get_rdataset?
faithful?
faithful.title

faithful = faithful.data
faithful.head()

faithful.shape

plt.hist(faithful.waiting)
plt.xlabel('Waiting time to next eruption (in mins)')
plt.ylabel('Frequency')
plt.title('Old Faithful Geyser time between eruption')
plt.show()

X = faithful.waiting
y = faithful.eruptions
model = sm.OLS(y, X)

results = model.fit()
results.summary()

print results.summary()

X = sm.add_constant(X)
X.head()

modelW0 = sm.OLS(y, X)
resultsW0 = modelW0.fit()
print resultsW0.summary()

newX = np.array([1,75])
resultsW0.params[0]*newX[0] + resultsW0.params[1] * newX[1]

resultsW0.predict(newX)

plt.scatter(faithful.waiting, faithful.eruptions)
plt.xlabel('Waiting time to next eruption (in mins)')
plt.ylabel('Eruption time (in mins)')
plt.title('Old Faithful Geyser')

plt.plot(faithful.waiting, resultsW0.fittedvalues, color='red', linewidth=3)
plt.show()

resids = resultsW0.resid

plt.plot(faithful.waiting, resids, 'o')
plt.hlines(y = 0, xmin = 40, xmax = 100)
plt.xlabel('Waiting time')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

print np.mean((faithful.eruptions - resultsW0.predict(X)) ** 2)

X = sm.add_constant(faithful.waiting)
y = faithful.eruptions

beta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
print "Directly estimating beta:", beta
print "Estimating beta using statmodels: ", resultsW0.params.values

from IPython.display import Image as Im
from IPython.display import display
Im('./images/shuttle.png')

data=np.array([[float(j) for j in e.strip().split()] for e in open("./data/chall.txt")])
data

from statsmodels.formula.api import logit, glm, ols

dat = pd.DataFrame(data, columns=["Temperature", "Failure"])
logit_model = logit("Failure ~ Temperature", dat).fit()
print logit_model.summary()

x = np.linspace(50, 85, 1000)
p = logit_model.params
eta = p["Intercept"] + x*p["Temperature"]
y = np.exp(eta)/(1 + np.exp(eta))

temps, pfail = data[:,0], data[:,1]
plt.scatter(temps, pfail)
axes=plt.gca()
plt.xlabel('Temperature')
plt.ylabel('Failure')
plt.title('O-ring failures')

plt.plot(x, y)

plt.xlim(50, 85)
plt.ylim(-0.1, 1.1)
