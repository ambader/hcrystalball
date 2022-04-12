import numpy as np
import pandas as pd

import hcrystalball

from hcrystalball.wrappers import get_sklearn_wrapper
from sklearn.linear_model import LinearRegression

from hcrystalball.utils import generate_tsdata
from hcrystalball.wrappers import ProphetWrapper

X, y = generate_tsdata(n_dates=365*2)
X_train, y_train, X_test, y_test = X[:-10], y[:-10], X[-10:], y[-10:]

model = ProphetWrapper()
y_pred = model.fit(X_train,y_train).predict(X_test)

model_lr = get_sklearn_wrapper(LinearRegression, lags=10)
mlr = model_lr.fit(X[:-10], y[:-10])

ar_ord = 2
hz = 10
train_me = np.array([0.,1.,0.,1.,0.,1.,0.,1.,0.,1.])
train_me=np.tile(train_me,100)
y = train_me
date_index = pd.date_range(start="2020-01-01", periods=len(y), freq="d")
X = pd.DataFrame(index=date_index)

for i in range(1,8):
    model_lry = get_sklearn_wrapper(LinearRegression, lags=ar_ord, fit_intercept=False,optimize_for_horizon=False)
    print("model_lry.predict(X[-9:-"+str(i)+":])")
    model_lry.fit(X[:-11],y[:-11])
    print(model_lry.predict(X[-9:-i]))
    print(model_lry.model.coef_)
