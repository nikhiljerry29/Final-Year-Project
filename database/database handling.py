import pandas as pd
# # Reading dataset
# s = pd.read_csv('dataset_sorghum.csv')
# p = pd.read_csv('dataset_pearl_millet.csv')
# # r = pd.read_csv('dataset_rice.csv')

# # Merging Dataset
# df = pd.merge(s, p, how = 'outer')
# # df = pd.merge(df, r, how = 'outer')


# # Randomizing Dataset
# df = df.sample (frac = 1)
# df = df.sample (frac = 1)
# df = df.sample (frac = 1)
# df = df.sample (frac = 1)
# df = df.sample (frac = 1)
# df = df.sample (frac = 1)
# df = df.sample (frac = 1)
# df = df.sample (frac = 1)

# # Writing Dataset
df = pd.read_csv('dataset.csv')
df = df.dropna()
print(len(df))
df.to_csv("dataset.csv",index = False)
