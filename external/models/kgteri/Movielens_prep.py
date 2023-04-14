def warn(*args, **kwargs):
    pass
from sklearn.model_selection import train_test_split

import warnings

warnings.warn = warn
import pandas as pd

public_df = pd.read_csv('//data/movielens/dataset.tsv', delimiter='\t', header=None)

#Create a user grouped dataframe
df3 = pd.DataFrame()
i=1
while df3.value_counts().size < 10000:
    df3 = df3.append(public_df.groupby(0).get_group(i), ignore_index=True)
    i=i+1

df = df3.sample(n=10000, random_state=43, axis=0).reset_index(drop=True)

# saving as tsv file
df.to_csv('//data/movielens/dataset2.tsv',header=False, index=False, sep="\t")

train, test = train_test_split(df, test_size=0.2)

train, val = train_test_split(train, test_size=0.1)

test = test.reset_index(drop=True)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

test.to_csv('//data/movielens/test.tsv',header=False, index=False, sep="\t")
train.to_csv('//data/movielens/train.tsv',header=False, index=False, sep="\t")
val.to_csv('//data/movielens/val.tsv',header=False, index=False, sep="\t")