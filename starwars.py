# %%
import pandas as pd

df = pd.read_parquet('dados/dados_clones.parquet')
df.head()

# %%
target = 'Status '
features = ['Estatura(cm)', 'Massa(em kilos)']
y = df[target]
print(y)
X = df[features]
print(X)

# %%
import sklearn.tree as tree
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)
# %%
import matplotlib.pyplot as plt
plt.figure(dpi=400)
tree.plot_tree(arvore, feature_names=X.columns, class_names=arvore.classes_, filled=True, max_depth=3)
plt.show()


# %%
df.groupby('Status ')[features].mean()
# %%
