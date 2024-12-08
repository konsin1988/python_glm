import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import statsmodels.formula.api as sf 
import numpy as np 

# survived
# sex
# Age
# Pclass

df = pd.read_csv('titanik_full_data_1.csv', sep='\t')

# Распределение выживших/погибших:
# sns.countplot(x= 'Survived', data=df)
# plt.xlabel('Survived?')
# plt.ylabel('Amount')
# plt.title('Fate of Titanik passengers')
# plt.show()

# Применим логистическую регрессию
logit_res = sf.glm('Survived ~ C(Pclass) + C(Sex) + Age', df, family = sm.families.Binomial()).fit()
print(logit_res.summary())