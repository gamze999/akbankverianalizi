#%%

import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
#%%

df = pd.read_csv("StudentsPerformance.csv")
df
#%%
def check_df(dataframe):
    print("################# shape ##################")
    print(dataframe.shape)
    print("################# types ##################")
    print(dataframe.dtypes)
    print("################# head ##################")
    print(dataframe.head())
    print("################# tail ##################")
    print(dataframe.tail())
    print("################# NA ##################")
    print(dataframe.isnull().sum())
    print("################# Quantiles ##################")
    print(dataframe.describe([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df)

df.describe()
#%%
df.info()

def grab_col_names(dataframe: pd.DataFrame, cat_th: int = 10, car_th: int = 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                    dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]


    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print("\n##################### Categoric #####################")
    print(cat_cols)
    print("\n##################### Numeric #####################")
    print(num_cols)
    print("\n##################### Categoric But Cardinal #####################")
    print(cat_but_car)
    print("\n##################### Numeric But Categoric #####################")
    print(num_but_cat)
    print("\n##################### Columns Overview #####################")
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


def grab_cols(dataframe, cat_th=10, car_th=20):
    """
    veri setindeki kategorik, numerik ve kategorik fakat kardinal olan değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri.

    Returns
    -------
    cat_cols: list
        kategorik değişken listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    -----
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"observations: {dataframe.shape[0]}")
    print(f"variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_cols(df)

def cat_summary(dataframe, col_name, plot = True):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio":100 * dataframe[col_name].value_counts() / len(df)})),
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

def hist_num_columns_show(dataframe, num_column):
    dataframe[num_column].hist(bins=20)
    plt.xlabel(num_column)
    plt.ylabel("Frequency")
    plt.show(block=True)


for num_col in num_cols:
    hist_num_columns_show(df, num_col)

#%%
plt.figure(figsize=(15,10))
ax = sns.barplot(x="math score", y="race/ethnicity", data=df)
plt.xticks(rotation= 90);
plt.show()

plt.figure(figsize=(15,10))
ax = sns.barplot(x="race/ethnicity", y="reading score", data=df)
plt.xticks(rotation= 90);
plt.show()

plt.figure(figsize=(15,10))
ax = sns.barplot(x="race/ethnicity", y="writing score", data=df)
plt.xticks(rotation= 90);
plt.show()

plt.figure(figsize=(15,10))
ax = sns.barplot(x="gender", y="math score", data=df)
plt.xticks(rotation= 90);
plt.show()

plt.figure(figsize=(15,10))
ax = sns.barplot(x="gender", y="reading score", data=df)
plt.xticks(rotation= 90);
plt.show()

plt.figure(figsize=(15,10))
ax = sns.barplot(x="gender", y="writing score", data=df)
plt.xticks(rotation= 90);
plt.show()

plt.figure(figsize=(15,10))
plt.scatter('race/ethnicity',"math score" , data=df)
plt.xticks(rotation=90)
plt.xlabel('race/ethnicity')
plt.ylabel('math score')
plt.show()

#%%
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu", annot=True)
plt.show()

#%%
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df,"math score")
outlier_thresholds(df,"writing score")
outlier_thresholds(df,"reading score")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,"math score")
check_outlier(df,"writing score")
check_outlier(df,"reading score")

#%%
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) |(dataframe[col_name] > up ))].index
        return outlier_index

grab_outliers(df, "math score")

#%%
df.isnull().sum()