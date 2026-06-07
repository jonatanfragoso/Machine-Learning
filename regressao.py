# %%

import pandas as pd

df = pd.read_excel('dados/dados_cerveja_nota.xlsx')
df.head()

# %%

from sklearn import linear_model

X = df[['cerveja']]
y = df['nota']

regressao = linear_model.LinearRegression(fit_intercept=True)

regressao.fit(X, y)


# %%
a, b = regressao.intercept_, regressao.coef_[0]
print(a,b)

# %%
prediction = regressao.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

plt.plot(X, y, 'o')
plt.grid(True)
plt.title('Relação entre cerveja e nota')
plt.xlabel('Cerveja')
plt.ylabel('Nota')

plt.plot(X.drop_duplicates(), prediction)
plt.legend(['Observado', 'y = {a:.3f} + {b:.3f} * x'.format(a=a, b=b)])
# %%
