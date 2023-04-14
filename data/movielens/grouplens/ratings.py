import pandas as pd

df3 = pd.read_csv('/home/klajdi/PycharmProjects/KGTERI/data/movielens/grouplens/ratings.dat',  sep = '::',
                   engine = 'python')
df4 = pd.DataFrame()
i=1
while i < 104:
    df4 = df4.append(df3.groupby("1").get_group(i), ignore_index=True)
    i=i+1
import numpy as np

myCsv = df4.astype(str).apply(lambda x: '::'.join(x), axis=1)
myCsv.rename('::'.join(df4.columns)).to_csv('/home/klajdi/PycharmProjects/KGTERI/data/movielens/grouplens/ratings2.dat', header=True,index=False)



