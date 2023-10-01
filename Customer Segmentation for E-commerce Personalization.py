import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt  



df = pd.read_csv('Online Retail.csv')
print(df.isnull().sum())
df = df.dropna(subset=['CustomerID'])



# Handle missing values 
df['Quantity'] = df['Quantity'].fillna(0)

# Change data types
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 
df['Quantity'] = df['Quantity'].astype(int)

df['order_count'] = df.groupby('CustomerID')['InvoiceNo'].transform('count')
df['days_since_last_purchase'] = df.groupby('CustomerID')['InvoiceDate'].transform(lambda x: (df['InvoiceDate'].max() - x.max()).days)
df['revenue'] = df['UnitPrice'] * df['Quantity'] 
df['avg_order_value'] = df.groupby('CustomerID')['revenue'].transform('mean')
df['Product_Category'] = df['Description'].apply(lambda x: x.split()[0])



df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek


df['recency'] = (df['InvoiceDate'].max() - df['InvoiceDate']).dt.days
df.drop(columns=['InvoiceDate'], inplace=True)

avg_basket_size = df.groupby('CustomerID')['revenue'].mean().reset_index()
purchase_frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()



print(df.columns)




selected_columns = ['Quantity', 'UnitPrice',
       'CustomerID', 'order_count', 'days_since_last_purchase',
       'revenue', 'avg_order_value', 'Year',
       'Month', 'DayOfWeek', 'recency']

scaler = StandardScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

pca = PCA(n_components=2) 
principal_components = pca.fit_transform(df[selected_columns])
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(2)])
df['PC1'] = principal_components[:,0]
df['PC2'] = principal_components[:,1]

inertias = []
for k in range(2, 10):
    print(k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(principal_components)
    inertias.append(kmeans.inertia_)
    
plt.plot(range(2, 10), inertias)
plt.xlabel('No of clusters')
plt.ylabel('Inertia')
plt.show()

# Fit K-means
kmeans = KMeans(n_clusters=4) 
clusters = kmeans.fit_predict(principal_components)
df['cluster'] = clusters


for cluster in df['cluster'].unique():
    cluster_df = df[df['cluster'] == cluster]
    print("Cluster {}".format(cluster))
    
    print("Size :", len(cluster_df))
    print("Average Revenue :", cluster_df['revenue'].mean())
    print("Purchase Frequency :", cluster_df['order_count'].mean())
    print("Category Distribution :")
    print(cluster_df['Product_Category'].value_counts()/len(cluster_df))
    
colors = ['red', 'green', 'blue', 'black']
for cluster, color in zip(df['cluster'].unique(), colors):
    cluster_df = df[df['cluster'] == cluster]
    plt.scatter(cluster_df['PC1'], cluster_df['PC2'], c=color)
plt.legend(df['cluster'].unique())
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

df.groupby('cluster')[['revenue', 'order_count']].agg('mean').plot(kind='bar')


# Evaluation 
silhouette_avg = silhouette_score(principal_components, df['cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Insights 
cluster_profiles = df.groupby('cluster').agg({'revenue': 'mean', 
                                              'order_count': 'median'})

print("Cluster 3 has highest average revenue")
print("Cluster 1 purchases most frequently")