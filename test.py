# Test code to test ML methods on housing dataset
# Author: Debotyam Maity
# Date: 09/28/2025

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset
dataset = pd.read_excel("./data/HousePricePrediction.xlsx")
print(dataset.head(5))