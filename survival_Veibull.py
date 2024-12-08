import pandas as pd 
import numpy as np 
import statsmodels.formula.api as sf
import statsmodels.api as sm 
import lifelines as lf 
import seaborn as sns
import matplotlib.pyplot as plt 

churn = pd.read_csv('Telco-Customer-Churn.csv')
print(churn.head(10))

# churn - ушёл клиент или нет
# tenure - сколько месяцев пробыл с компанией
# SeniorCitizen - пожилой?
# Dependents - наличие иждивенцев в семье
# MonthlyCharged - оплата в месяц
# PaperlessBilling - оплата с чеком или без

# sns.countplot(x="Churn", data=churn)
sns.displot(churn.tenure, kde=False)
plt.xlabel('Month count')
plt.ylabel('Frequency')
plt.title('How many months clients have been in the company?')
# plt.show()

# Подготовка данных
churn.tenure = churn.tenure + 0.001 # without zero months 
# Функция связи - логарифм, а логарифм от нуля брать нельзя

churn.Churn = np.where(churn.Churn == 'Yes', 1, 0) #yes = 1, no = 0
churn.SeniorCitizen = np.where(churn.SeniorCitizen == 1, 'Yes', 'No') #наоборот

surv = lf.WeibullAFTFitter()
surv.fit(df = churn, duration_col='tenure', event_col='Churn', formula='C(SeniorCitizen) + C(Dependents) + MonthlyCharges + C(PaperlessBilling)')
surv.print_summary()