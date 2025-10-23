# Central Tendancy and Dispersion Measures

# Mean, median and mode describe the center of the data. Variance and standard deviation show how spread out the data is.

import pandas as pd
import numpy as np


# Example: synthetic daily-sales data (replace with your CSV)
sales = pd.Series([1200, 1150, 1300, 1250, 1190, 1350, 1280, 1200, 0, np.nan])

# Handling missing: drop or fill
sales_clean = sales.dropna()

mean = sales_clean.mean()
median = sales_clean.median()
mode = sales_clean.mode().tolist()  # could be multiple
variance = sales_clean.var(ddof=0)  # population
stddev = sales_clean.std(ddof=0)

print("Mean:", mean)
print("Median:", median)
print("Mode(s):", mode)
print("Variance:", variance)
print("StdDev:", stddev)
