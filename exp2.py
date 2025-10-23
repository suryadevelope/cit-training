# Preprocessing: Feature selection, Missing values, Discretization, Outliers
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats

# Example dataset (replace with pd.read_csv('file.csv'))
df = pd.DataFrame({
    'temp': [30, 31, np.nan, 29, 300],   # 300 is outlier
    'humidity': [45, 50, 49, np.nan, 48],
    'vibration': [0.1, 0.2, 0.15, 0.14, 0.13],
    'label': [0,1,1,0,1]
})

# a) Attribute selection (select top 2)
X = df[['temp','humidity','vibration']]
y = df['label']
# Impute temporarily for selection
imp = SimpleImputer(strategy='mean')
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
selector = SelectKBest(score_func=f_classif, k=2)
selector.fit(X_imp, y)
selected_cols = X.columns[selector.get_support()].tolist()
print("Selected attributes:", selected_cols)

# b) Handling missing values
imputer = SimpleImputer(strategy='mean')
X_filled = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# c) Discretization (e.g., temp into 3 bins)
kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
df['temp_binned'] = kb.fit_transform(X_filled[['temp']]).astype(int).reshape(-1)

# d) Outlier elimination (z-score filter)
z = np.abs(stats.zscore(X_filled))
mask = (z < 3).all(axis=1)  # keep rows where all features have |z|<3
df_clean = df[mask].reset_index(drop=True)
print("After outlier removal, rows:", len(df_clean))
print(df_clean)

