import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df_country0 = pd.read_csv(r"C:\Users\moham\Desktop\MY AI\Projects\Machine Learning Projects\Country Data\Country-data.csv")
df_country0 = pd.DataFrame(df_country0)


# print(df_country.head(5))
# print(df_country.shape)
# print(df_country.columns)
# print(df_country.dtypes)
# print(df_country.info())
# print(df_country.isnull().sum())
# print(df_country.describe())
# print(df_country.duplicated().sum())

df_country = df_country0.drop(columns=['country'])



cor_matrix = df_country.corr
# print(cor_matrix)
# fig, ax = plt.subplots() 
# fig.set_size_inches(15,10)
# sns.heatmap(cor_matrix, vmax =.8, square = True, annot = True,cmap='YlGn' )
# plt.title('Correlation Matrix',fontsize=15)
# plt.show()

# Most related features :
# income / Gdp : 0.9 -> highly positive correlated
# child_mort / life_expect : -0.89 -> highly negative correlated
# total_fer / child_mort : 0.85 -> highly positive correlated


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_country = pd.DataFrame(scaler.fit_transform(df_country), columns=df_country.columns)

from sklearn.decomposition import PCA, IncrementalPCA
pca = PCA(n_components=9).fit(df_country)

pca = PCA(n_components=9).fit(df_country)
exp = pca.explained_variance_ratio_
# print(exp)

# plt.plot(np.cumsum(exp), linewidth=2, marker = 'o', linestyle = '--')
# plt.title("PCA")
# plt.xlabel('n_component')
# plt.ylabel('Cumulative explained Variance Ratio')
# plt.yticks(np.arange(0.5, 1.05, 0.05))
#  plt.show()

finla_pca = IncrementalPCA(n_components=5).fit_transform(df_country)
pc = np.transpose(finla_pca)

df = pd.DataFrame({
    'PC1':pc[0],
    'PC2':pc[1],
    'PC3':pc[2],
    'PC4':pc[3],
    'PC5':pc[4],
})
# print(df_country.sample(5))


from sklearn.cluster import KMeans
inertias = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_country)
    inertias.append(kmeans.inertia_)

# # Plot the elbow curve
# plt.plot(range(1, 11), inertias, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.show()

kmeans = KMeans(n_clusters=3).fit(df)
df.insert(0, 'Country', df_country0['country'])
df['class'] = kmeans.labels_
df['Label'] = df['class']
# print(df.sample(5))

poor = int(df[df.Country=='Afghanistan']['class'])
midle = int(df[df.Country=='Iran']['class'])
rich = int(df[df.Country=='Canada']['class'])

poor_label = 'Poor countries'
midle_label = 'Midle countries'
rich_label = 'Rich countries'


df.replace({'Label':{poor:'Poor countries', midle:'Midle countries', rich:'Rich countries'}},inplace=True)
print(df.sample(6))

import plotly.express as px
fig = px.choropleth(df[['Country','class']],
                    locationmode = 'country names',
                    locations = 'Country',
                    color = df['Label'],  
                    color_discrete_map = {'Rich countries': 'Green',
                                          'Midle countries':'LightBlue',
                                          'Poor countries':'Red'}
                   )

fig.update_layout(
        margin = dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=2,
            ),
    )
fig.show()



