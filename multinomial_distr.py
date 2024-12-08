import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.formula.api as sf 

df = pd.read_csv('titanik_full_data_1.csv', sep='\t')

sns.countplot(x='Pclass', data=df)
plt.show()

multi_res = sf.mnlogit('Pclass ~ C(Sex) + Age', df).fit()
print(multi_res.summary())