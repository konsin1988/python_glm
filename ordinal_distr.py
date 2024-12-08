import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.formula.api as sf 
import numpy as np 

from bevel.linear_ordinal_regression import OrderedLogit
wines = pd.read_csv('winequality-red.csv', sep=';')

sns.countplot(x='quality', data=wines)
plt.show()

#bevel не поддерживает формулы, отдельно сохраняем зависимые и независимые переменные
y = wines.quality
x = wines.drop('quality', axis=1)
ol = OrderedLogit()
ol.fit(x, y)
ol.print_summary()