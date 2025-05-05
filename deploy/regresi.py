import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("company_esg_financial_dataset.csv")

print(df.head())
print(df.describe())

cdf = df[['ESG_Overall','ESG_Environmental','ESG_Social','ESG_Governance',
          'Revenue','ProfitMargin','MarketCap','CarbonEmissions', 'EnergyConsumption']]

viz = df[['Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
          'ESG_Overall', 'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
          'CarbonEmissions', 'WaterUsage', 'EnergyConsumption']]
viz.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

for col in ['ESG_Governance', 'ESG_Environmental', 'ESG_Social']:
    plt.scatter(df['ESG_Overall'], df[col], color='blue')
    plt.xlabel("ESG_Overall")
    plt.ylabel(col)
    plt.title(f"ESG_Overall vs {col}")
    plt.show()

X = cdf[['ESG_Overall']].values
y = cdf[['ESG_Governance']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_scaled, y_scaled.ravel())

esg_input = 100
y_pred_scaled = regressor.predict(sc_X.transform(np.array([[esg_input]])))
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
print(f"Prediksi ESG_Governance untuk ESG_Overall={esg_input}: {y_pred[0][0]:.2f}")

X_grid = np.arange(min(X_scaled), max(X_scaled), 0.01).reshape(-1, 1)
plt.scatter(X_scaled, y_scaled, color='blue')
plt.plot(X_grid, regressor.predict(X_grid), color='red')
plt.title('ESG_Overall vs ESG_Governance (scaled)')
plt.xlabel('ESG_Overall (scaled)')
plt.ylabel('ESG_Governance (scaled)')
plt.show()

plt.scatter(sc_X.inverse_transform(X_scaled), sc_y.inverse_transform(y_scaled), color='blue')
plt.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(regressor.predict(X_grid).reshape(-1, 1)), color='red')
plt.title('ESG_Overall vs ESG_Governance (original scale)')
plt.xlabel('ESG_Overall')
plt.ylabel('ESG_Governance')
plt.show()


plt.figure(figsize=(20, 10))
tree.plot_tree(regressor.estimators_[0], feature_names=['ESG_Overall'], filled=True)
plt.show()

X2 = cdf[['EnergyConsumption']]
y2 = cdf[['CarbonEmissions']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

rf2 = RandomForestRegressor(n_estimators=100, random_state=0)
rf2.fit(X2_train, y2_train.values.ravel())
y2_pred = rf2.predict(X2_test)

print("========== R² Score Output ==========")
print("Mean Absolute Error:", mean_absolute_error(y2_test, y2_pred))
print("Mean Squared Error:", mean_squared_error(y2_test, y2_pred))
print("R² Score:", r2_score(y2_test, y2_pred))