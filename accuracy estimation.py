import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', rc = {'figure.figsize' : (10 , 5)})

df = pd.read_csv('database/accuracyvsepochs3.csv')
df['Total Neurons'] = df['First Layer Neurons'] + df['Second Layer Neurons']
df = df.sort_values('Accuracy')
print(df[['Accuracy','Total Neurons']])
sns.scatterplot(x = 'First Layer Neurons', y = 'Accuracy', data = df)
plt.show()
# print(df[df['Accuracy'] == df['Accuracy'].max()])