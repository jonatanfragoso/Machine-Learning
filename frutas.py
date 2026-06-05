# %%

import pandas as pd

df = pd.read_excel('dados/dados_frutas.xlsx')
df
# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)


# %%
y = df['Fruta']
features = ['Arredondada', 'Vermelha', 'Suculenta', 'Doce']
X = df[features]
# %%

arvore.fit(X, y)
# %%
arvore.predict([[0, 0, 0, 0]])
# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400, figsize=(10, 10))
tree.plot_tree(arvore, feature_names=features, class_names=arvore.classes_, filled=True)
plt.show()

# %%
proba = arvore.predict_proba([[1, 1, 1, 1]])[0]
pd.Series(proba, index=arvore.classes_)