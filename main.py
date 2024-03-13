import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
data = pd.read_csv("C:\\Users\\mumtaz\\OneDrive\\Desktop\\pandas\\bengaluru_house_prices.csv")
data.head()
data.shape
data.info()
for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)

data.isna().sum()
data.drop(columns=['area_type', 'availability', 'society', 'balcony'],inplace=True)
data.describe()
data.info()
data['location'].value_counts()
data['size'].value_counts()
data['location'] = data['location'].fillna('Sarjapur Road')
data['size'].value_counts()
data['size'] = data['size'].fillna('2 BHK')
data['size'] = data['bath'].fillna(data['bath'].median())
data.info()

data['size'] = data['size'].astype(str)
data['bhk'] = data['size'].str.split().str.get(0).astype(float)

data = data[data['bhk'] <= 20]
data['total_sqft'].unique()
def convertRange(x):
    if '_' in str(x):
        temp = x.split('_')
        return(float(temp[0]) + float(temp[1]))/2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convertRange)
data.head()

data['price_per_sqft']= data['price']*100000 / data['total_sqft']
data['price_per_sqft']
data.describe()

data['location'].value_counts()

data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()

location_count_less_10 = location_count[location_count <= 10]
location_count_less_10

data['location'] = data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)
data.describe()

(data['total_sqft']/ data['bhk']).describe()
data = data[((data['total_sqft']/data['bhk']) >= 300 )]
data.shape
data.price_per_sqft.describe()


def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        gen_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output

data = remove_outliers_sqft(data)

def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    bhk_stats = {}
    for location, location_df in df.groupby('location'):
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }

    for bhk, bhk_df in data.groupby('bhk'):
        stats = bhk_stats.get(bhk - 1)
        if stats and stats['count'] > 5:
            exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < (stats['mean'])].index.values)

    return df.drop(exclude_indices, axis='index')

data = bhk_outlier_remover(data)

data.drop(columns=['size', 'price_per_sqft'], inplace=True)
data.head()

data.to_csv("Cleaned_data.csv")
X=data.drop(columns=['price', 'location'])
y=data['price']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)

column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown= 'ignore'), ['location']), remainder='passthrough'
)

scaler = StandardScaler()
lr = LinearRegression()
pipe = make_pipeline(column_trans, scaler, lr)

pipe.fit(X_train,y_train)

y_pred_lr= pipe.predict(X_test)
r2_score(y_test, y_pred_lr)


lasso = Lasso()
pipe = make_pipeline(column_trans, scaler, lasso)
pipe.fit(X_train, y_train)
y_pred_lasso = pipe.predict(X_test)
r2_score(y_test, y_pred_lasso)



ridge = Ridge()
pipe = make_pipeline(column_trans, scaler, ridge)
pipe.fit(X_train, y_train)
y_pred_ridge= pipe.predict(X_test)
r2_score(y_test, y_pred_ridge)


print("No Regularization:", r2_score(y_test, y_pred_lr))
print("No Lasso:", r2_score(y_test, y_pred_lasso))
print("Ridge:", r2_score(y_test, y_pred_ridge))


pickle.dump(pipe, open('RidgeModel.pkl', 'wb'))





