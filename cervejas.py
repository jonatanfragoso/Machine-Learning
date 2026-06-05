# %%   

import pandas as pd

df = pd.read_excel('dados/dados_cerveja.xlsx')
df.head()

# %%
target = 'classe'
features = ['temperatura', 'copo', 'espuma', 'cor']
y = df[target]
X = df[features]

X.replace({'mud': 1, 'pint': 2, 'sim': 1, 'não': 0, 'clara': 0, 'escura': 1}, inplace=True)

# %%

import sklearn.tree as tree
arvore = tree.DecisionTreeClassifier(random_state=42)

arvore.fit(X, y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(arvore, feature_names=X.columns, class_names=arvore.classes_, filled=True)
plt.show()
