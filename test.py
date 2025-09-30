# Test code to test ML methods on housing dataset
# Author: Debotyam Maity
# Date: 09/28/2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score

# import dataset
dataset = pd.read_excel("./data/HousePricePrediction.xlsx")
pd.set_option('display.max_columns', None)
# display data information
print(dataset.info())

obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)

# EDA to identify patterns and anomalies - seaborn (heatmap) for numerical data
numerical_datasubset = dataset.select_dtypes(include=['number'])
fig = plt.figure(figsize=(10, 5))
sns.heatmap(numerical_datasubset.corr(),
            cmap = "RdYlGn",
            fmt = ".2f",
            linewidths = 1,
            annot = True)
fig.savefig('./outputs/correlation_hm.png')
plt.close(fig)

# EDA to look at categorical features
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)

data = {'Objects': object_cols, 'Values': unique_values}
df = pd.DataFrame(data)
fig = plt.figure(figsize=(10, 5))
plt.title('No. unique values (categorical features)')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values, hue='Values', data=df, palette='viridis')
fig.savefig('./outputs/unique_b.png')

# Exterior1st has 16 unique categorical values. MSZoning has 6, LotConfig and BldgType have 5 each.

fig = plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    data = {'Objects': list(y.index), 'Values': y}
    df = pd.DataFrame(data)
    sns.barplot(x=list(y.index), y=y, hue='Values', data=df, palette='viridis')
    index += 1
    ax = plt.gca()
    ax.margins(x=0.01, y=0.01)

fig.savefig('./outputs/categories_b.png')
plt.close(fig)

# Distribution of these categorical values (features) separately for each of these categories suggests some values are
# more dominant and some are very few in numbers.
# Next we should do data cleaning to remove columns that are not required for model prediction.

# remove Id column
dataset.drop(['Id'], axis=1, inplace=True)

# SalePrice has empty values. We can replace them with mean values to make the distribution symmetric instead of skewed.
# dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
# This has unintended consequence of creating single value cluster at the mean price point. Instead we can remove
# the entries with missing SalePrice values
dataset.dropna(subset=['SalePrice'], inplace=True)

# drop all null values from dataset
new_dataset = dataset.dropna()

# verify that there are no null valuies in the cleaned dataset
print(new_dataset.isnull().sum())

# categorical data needs to be converted to binary vectors. We can use one hot encoding to convert the categorical
# values to integers.
# collect all features which have object datatype
ff = (new_dataset.dtypes == 'object')
object_cols = list(ff[ff].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', len(object_cols))
# apply OneHotEncoding to this list
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)
print(df_final.info())

# split dataset into training and testing datasets
# dataset split to Sale Price as prediction variable and other features as inputs
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# apply ML models to cleaned dataset
# loss function will be mean absolute error
# SVM model
model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_svm = model_SVR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred_svm))

# RFG model
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred_rfr = model_RFR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred_rfr))

# LR model
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_lr = model_LR.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred_lr))

# XGBoost model
model_XGB = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
model_XGB.fit(X_train, Y_train)
Y_pred_xgb = model_XGB.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred_xgb))

# XGBoost provides the best estimate followed by SVM. However, if we remove the entries with missing SalePRice data,
# XGBoost has best estimates followed by random forest
# apply bagging to see if prediction can be improved
#regr = BaggingRegressor(estimator=svm.SVR(), n_estimators=10, random_state=0).fit(X_train, Y_train)
regr = BaggingRegressor(estimator=RandomForestRegressor(), n_estimators=10, random_state=0).fit(X_train, Y_train)
#regr = BaggingRegressor(estimator=xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3), n_estimators=10, random_state=0).fit(X_train, Y_train)
Y_pred_bag = regr.predict(X_valid)
print(mean_absolute_percentage_error(Y_valid, Y_pred_bag))

# bagging improves SVM prediction only slighly. This suggests that it might make sense to use the base predictor model
# instead of bagging.
slope_svm, intercept_svm = np.polyfit(Y_valid, Y_pred_svm, 1)
slope_rfr, intercept_rfr = np.polyfit(Y_valid, Y_pred_rfr, 1)
slope_lr, intercept_lr = np.polyfit(Y_valid, Y_pred_lr, 1)
slope_xgb, intercept_xgb = np.polyfit(Y_valid, Y_pred_xgb, 1)
slope_bag, intercept_bag = np.polyfit(Y_valid, Y_pred_bag, 1)
svm_fit = slope_svm * Y_valid + intercept_svm
rfr_fit = slope_rfr * Y_valid + intercept_rfr
lr_fit = slope_lr * Y_valid + intercept_lr
xgb_fit = slope_xgb * Y_valid + intercept_xgb
bag_fit = slope_bag * Y_valid + intercept_bag

# SVM has very poor performance so we drop it from our visualization
# Applying bagging technique to Random Forest Regressor instead gives good results (almost as good as XGBoost).

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(Y_valid.tolist(), Y_pred_xgb.tolist(), s=5, c='g')
ax1.plot(Y_valid, xgb_fit, color='b', label='XGBoost Best Fit')
ax1.set_xlabel("Reported Price")
ax1.set_ylabel("Predicted Price")
ax1.set_title('XGBoost')
ax1.tick_params(axis='x', labelrotation=45)
ax1.ticklabel_format(useOffset=False, style='plain')
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(Y_valid.tolist(), Y_pred_rfr.tolist(), s=5, c='k')
ax2.plot(Y_valid, rfr_fit, color='k', label='RF Best Fit')
ax2.set_xlabel("Reported Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title('Random Forest')
ax2.tick_params(axis='x', labelrotation=45)
ax2.ticklabel_format(useOffset=False, style='plain')
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(Y_valid.tolist(), Y_pred_lr.tolist(), s=5, c='r')
ax3.plot(Y_valid, lr_fit, color='r', label='LR Best Fit')
ax3.set_xlabel("Reported Price")
ax3.set_ylabel("Predicted Price")
ax3.set_title('Linear')
ax3.tick_params(axis='x', labelrotation=45)
ax3.ticklabel_format(useOffset=False, style='plain')
ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(Y_valid.tolist(), Y_pred_bag.tolist(), s=5, c='b')
#ax4.plot(Y_valid, bag_fit, color='g', label='SVM - Bagging Best Fit')
ax4.plot(Y_valid, bag_fit, color='g', label='RF - Bagging Best Fit')
ax4.set_xlabel("Reported Price")
ax4.set_ylabel("Predicted Price")
ax4.set_title('Random Forest - Bagging')
ax4.tick_params(axis='x', labelrotation=45)
ax4.ticklabel_format(useOffset=False, style='plain')
plt.tight_layout()
fig.savefig('./outputs/compare_models_1.png')
plt.close(fig)

# generate residuals plot
fig = plt.figure()
data_residuals = pd.DataFrame({'X': Y_valid.tolist(), 'Y': Y_pred_xgb.tolist()})
sns.residplot(x='X', y='Y', data=data_residuals, lowess=True, line_kws={'color': 'g'})
data_residuals = pd.DataFrame({'X': Y_valid.tolist(), 'Y': Y_pred_rfr.tolist()})
sns.residplot(x='X', y='Y', data=data_residuals, lowess=True, line_kws={'color': 'k'})
data_residuals = pd.DataFrame({'X': Y_valid.tolist(), 'Y': Y_pred_lr.tolist()})
sns.residplot(x='X', y='Y', data=data_residuals, lowess=True, line_kws={'color': 'r'})
data_residuals = pd.DataFrame({'X': Y_valid.tolist(), 'Y': Y_pred_bag.tolist()})
sns.residplot(x='X', y='Y', data=data_residuals, lowess=True, line_kws={'color': 'b'})
plt.xlabel("Reported Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(0, color='grey', linestyle='--')  # Add a horizontal line at y=0
fig.savefig('./outputs/compare_models_2.png')
plt.close(fig)

# the residuals plot clearly shows poor predictive capability of models at higher prices, likely due to fewer data
# points. Also, random forest and XGBoost show similar performance and RF may be chosen doue to lower compute needs.
# Bagging does not improve performance to any significant extent for this dataset.
