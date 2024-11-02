# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries like pandas, numpy, sklearn, and matplotlib. Load the customer data (e.g., purchase behavior, demographics) and inspect it to understand the features. 
2. Handle missing values and normalize/scale the features using StandardScaler for better clustering performance. Ensure all data points are numeric and consistent, removing any outliers if necessary.
3. Use the Elbow Method or Silhouette Score to find the best value for 𝐾. Plot the sum of squared distances (inertia) for different values of K and choose the optimal K where the elbow appears.
4. Use KMeans from sklearn.cluster to fit the model to the preprocessed customer data.
Set the number of clusters K and initialize the model with random centroids to start the clustering process. 
5. Use the trained K-Means model to predict and assign each customer to one of the K clusters.
Each customer will now be labeled with a cluster ID, indicating their segment.
6. Use matplotlib or seaborn to create a 2D or 3D plot of the customer segments based on their feature values. This visualization helps interpret and analyze the distinct customer groups formed by the clustering algorithm.
7. Analyze the characteristics of each cluster, understanding the differences between customer segments. Use this information to tailor marketing strategies or improve customer engagement based on their unique behaviors.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: ABISHEK PV
RegisterNumber: 212222230003

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\admin\Downloads\Mall_Customers.csv")

data.head()
data.info()
data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []  # Within-Cluster Sum of Square
# It is the sum of squared distance between each point & the centroid in a cluster.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

km = KMeans(n_clusters=5)
km.fit(data.iloc[:, 3:])

KMeans(n_clusters=5)

y_pred = km.predict(data.iloc[:, 3:])
y_pred

data['cluster'] = y_pred
o = data[data['cluster'] == 0]
p = data[data['cluster'] == 1]
q = data[data['cluster'] == 2]
r = data[data['cluster'] == 3]
s = data[data['cluster'] == 4]

plt.scatter(o['Annual Income (k$)'], o['Spending Score (1-100)'], c="red", label="cluster0")
plt.scatter(p['Annual Income (k$)'], p['Spending Score (1-100)'], c="blue", label="cluster1")
plt.scatter(q['Annual Income (k$)'], q['Spending Score (1-100)'], c="green", label="cluster2")
plt.scatter(r['Annual Income (k$)'], r['Spending Score (1-100)'], c="orange", label="cluster3")
plt.scatter(s['Annual Income (k$)'], s['Spending Score (1-100)'], c="magenta", label="cluster4")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.title("Customer Segments")

*/ 
```

## Output:

# Data Info

![Screenshot 2024-10-17 205937](https://github.com/user-attachments/assets/86d6c5ac-3b5f-47a3-93d0-e59bbae59523)


# Null values 

![Screenshot 2024-10-17 205943](https://github.com/user-attachments/assets/1172882a-5433-48f4-98d7-301d1cd483ce)


# Plot

![Screenshot 2024-10-17 205953](https://github.com/user-attachments/assets/9c299df4-abca-4ee6-9e38-4ed3556a7d08)


# Array

![Screenshot 2024-10-17 210002](https://github.com/user-attachments/assets/b4561b60-ec9e-4897-828e-ebcf256c6fa1)


# Clustering

![Screenshot 2024-10-17 210012](https://github.com/user-attachments/assets/34c65461-f646-46dd-93ef-dcf3cec98f8e)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
