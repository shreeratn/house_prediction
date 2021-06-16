import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

data = pd.read_csv("C:/DjangoProjects/houseprice/train_.csv")
x = data.drop("SalePrice",axis=1)
y=data["SalePrice"]
model=ExtraTreesRegressor()
model.fit(x,y)
m = model.feature_importances_
a = x.columns
print(np.average(m))
for i in zip(a,m):
    print(i,end=",")
