import numpy as np
import matplotlib.pyplot as plt

data = np.array([2, 8, 15, 6, 3])

c1 = np.random.randint(min(data), max(data))
c2 = np.random.randint(min(data), max(data))
print('Initial Random Cluster1: ',c1)
print('Initial Random Cluster2: ',c2)

for i in range(10):
    cluster1,cluster2 = [],[]


    # for j in range(len(data)):
    #     if(abs(data[j]-c1) < abs(data[j]-c2)):
    #         cluster1.append(data[j])
    #         print('cluster1: ',cluster1)
    #     elif(abs(data[j]-c2) < abs(data[j]-c1)):
    #         cluster2.append(data[j])
    #         print('cluster2: ',cluster2)
    for x in data:
        if abs(x - c1) < abs(x - c2):
            cluster1.append(x)
        else:
            cluster2.append(x)
    # newc1 = np.mean(cluster1)
    # print(newc1)
    # newc2 = np.mean(cluster2)
    # print(newc2)
    newc1 = np.mean(cluster1) if cluster1 else c1
    newc2 = np.mean(cluster2) if cluster2 else c2

    if(newc1-c1 == 0 and newc2-c2 == 0):
        break

    #assign new centroids (after mean) to old centroids 
    c1,c2 = newc1, newc2

print('new centroid1: ',newc1)
print('new centroid2: ',newc2)
print("Final centroid1:", c1)
print("Final centroid2:", c2)
print("Cluster1 values:", cluster1)
print("Cluster2 values:", cluster2)

# == Plotting each clusters with datapoint and centroids ==
plt.figure(figsize=(10,6))

# Slight vertical offsets so clusters don’t overlap visually
plt.scatter(cluster1 ,[0.05]*len(cluster1), color='blue', s=120, label='Cluster 1')
plt.scatter(cluster2 ,[0.05]*len(cluster2), color='orange', s=120, label='Cluster 2')
# Label each data point with its value
for x in cluster1:
    plt.text(x, 0.051, str(x), ha='center', fontsize=10, color='blue')
for x in cluster2:
    plt.text(x, 0.051, str(x), ha='center', fontsize=10, color='orange')


# Plot centroids
plt.scatter(c1, 0.05, color='red', marker='x', s=200, label='Centroid 1')
plt.scatter(c2, 0.05, color='green', marker='x', s=200, label='Centroid 2')
# Label centroids with their mean values
plt.text(c1, 0.051, f'C1={c1:.2f}', ha='center', fontsize=10, color='red', fontweight='bold')
plt.text(c2, 0.051, f'C2={c2:.2f}', ha='center', fontsize=10, color='green', fontweight='bold')



# Formatting
plt.title('1D K-Means Clustering with Labeled Points')
plt.xlabel('Data Value')
plt.yticks([])
plt.legend()
plt.show()