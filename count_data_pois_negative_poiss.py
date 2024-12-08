import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.formula.api as sf 
import statsmodels.api as sm 
import seaborn as sns 
import numpy as np 

def string_sep():
    print('\n', '-' * 80 )
    print('-' * 80, '\n' )

credit = pd.read_csv('credit_card__1_.csv')

# Распределение количества активных счетов
sns.countplot(x='active', data=credit)
# plt.show()

# Построим 4 вида модели: Пуассон, Отрицательное биномиальное, С повышенным содержанием нулей
# Пуассон:
pois = sf.glm('active ~ age + income + expenditure + C(owner) + C(selfemp)', family = sm.families.Poisson(), data=credit).fit()
print('Poisson model\n')
print(pois.summary())
print(f'Overdispersion = {pois.pearson_chi2/pois.df_resid}')

# Отрицательное биномиальное распределение
neg = sf.glm('active ~ age + income + expenditure + C(owner) + C(selfemp)', data=credit, family=sm.families.NegativeBinomial(alpha=0.45)).fit()
# alpha отвечает за дисперсию, находится в промежутке (0.1; 2.0)
# подбирал опытным путём
string_sep()
print('\nNegative binomial model\n')
print(neg.summary())
print(f'Overdispersion = {neg.pearson_chi2/neg.df_resid}')
string_sep()

# Сравниваем две модели с помощью информационного критерия Акаике (AIC)
print(f'For Poisson model AIC = {pois.aic}\nFor Negative Binomial model AIC = {neg.aic}')
string_sep()

# Подготовка данных для работы с моделью с повышенным содержанием нулей
credit.owner = np.where(credit.owner == 'yes', 1, 0)
credit.selfemp = np.where(credit.selfemp == 'yes', 1, 0)

Y = credit.active
X = credit.loc[:, ['owner', 'selfemp', 'age', 'income', 'expenditure']]
X = sm.add_constant(X)

# ZeroInflatedPoisson. Чем сложнее модель, тем она капризнее. Увеличиваем количество итераций и меняем алгоритм на более стабильный
zeroinf = sm.ZeroInflatedPoisson(Y, X).fit(maxiter = 100, method = 'ncg')
print(zeroinf.summary())

string_sep()
print(f'For ZeroInflatedPoisson AIC = {zeroinf.aic}')
string_sep()

# ZeroInflatedNegativeBinomial
zeroinf_NB = sm.ZeroInflatedNegativeBinomialP(Y, X).fit(maxiter=100, method='ncg')
print(zeroinf_NB.summary())
string_sep()
print(f'For Poisson model AIC = {pois.aic}\nFor Negative Binomial model AIC = {neg.aic}')
print(f'For ZeroInflatedPoisson AIC = {zeroinf.aic}')
print(f'For ZeroInflatedNegativeBinomial AIC = {zeroinf_NB.aic}')
string_sep()