import joblib
import pandas as pd
from lightgbm import LGBMRegressor as lgbr

train = pd.read_csv('../data/train.csv')
X = train.drop('SalePrice', axis=1)
y = train.SalePrice
model = lgbr(bagging_fraction=0.8, bagging_freq=5, feature_fraction=0.5,
             min_child_samples=91, min_split_gain=0.9, n_estimators=300,
             num_leaves=60, random_state=937, reg_alpha=0.7, reg_lambda=0.7)
model.fit(X, y)
joblib.dump(model, '../model/model.pkl')