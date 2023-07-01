from app_store_scraper import AppStore
import pandas as pd
import numpy as np
import seaborn as sns


peloton = AppStore(country='us', app_name='peloton', app_id = '792750948')
peloton.review(how_many=25)
df = pd.DataFrame(np.array(peloton.reviews),columns=['review'])
df2 = df.join(pd.DataFrame(df.pop('review').tolist()))
df3 = df2.drop(['developerResponse','title' , 'isEdited' , 'userName' , 'review'], 1)
print(df3.head(5))
for col in df3.columns:
    print(col)
print()
print(df3.dtypes)
print()
print(df3.describe(include='all', datetime_is_numeric=True))
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    df3['rating'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='rating', ylabel='Count');

df3.loc[]