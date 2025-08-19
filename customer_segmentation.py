import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df= pd.read_csv('Mall_Customer_Dataset.csv')

st.title('Customer Segmentation Dashboard')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

for k in range(2,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)


cluster_ids =df['Cluster'].unique()
selected_cluster =st.sidebar.selectbox('Select Cluster', sorted(cluster_ids))

st.header(f'Cluster {selected_cluster} Summary')
cluster_data = df[df['Cluster']==selected_cluster]
st.write(cluster_data.describe())


st.subheader('Feature Distribution for Selected Cluster')
features =['Age','Annual Income (k$)', 'Spending Score (1-100)']
for feature in features:
    fig, ax = plt.subplots()
    sns.histplot(cluster_data[feature], kde=True, ax=ax)
    ax.set_title(f'{feature} Distribution (Cluster{selected_cluster})')
    st.pyplot(fig)

cluster_actions = {
    0:'Target with exclusive premium product promotions.',
    1:'Personalized engagement to increase spending (VIP Offers).',
    2:'Education about high-value products, loyalty incentives.',
    3:'Invite to special events, premium club memberships.',
    4:'Cross-sell related premium products.',
    5:'Referral programs for high-income peers.',
    6:'Personalized recommendations based on histoery.',
    7:'Early access to new products.',
    8:'Targeted email campaings with exclusive deals.',
    9:'High-touch customer service, request feedback.',
}

st.subheader('Recommended Action')
st.write(cluster_actions.get(selected_cluster,'No action defined for this Cluster.'))

if st.checkbox('Show Cluster Summary Table'):
    st.write(df.groupby('Cluster').mean(numeric_only=True))