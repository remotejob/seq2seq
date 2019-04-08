
import pandas as pd

df=pd.read_table('data/tmp.tsv', header=None)
df.head()

df.shape

df_subset=df.iloc[:1000,]

df_subset.to_csv('data/tmp_sub.txt', header=None, index=None, sep=' ')