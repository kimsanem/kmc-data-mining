import numpy as np
import matplotlib.pyplot as plt

data = np.array([2, 8, 15, 6, 3])

c1 = np.random.randint(min(data), max(data))
c2 = np.random.randint(min(data), max(data))
print('Initial Random Cluster1: ',c1)
print('Initial Random Cluster2: ',c2)

for i in range(10):
    cluster1 = []
    cluster2 = []

    for j in range(len(data)):
        if(abs(data[j]-c1) < abs(data[j]-c2)):
            cluster1.append(data[j])
            print('cluster1: ',cluster1)
        elif(abs(data[j]-c2) < abs(data[j]-c1)):
            cluster2.append(data[j])
            print('cluster2: ',cluster2)
    newc1 = np.mean(cluster1)
    print(newc1)
    newc2 = np.mean(cluster2)
    print(newc2)

    if(newc1-c1 == 0 and newc2-c2 == 0):
        break
    c1,c2 = newc1, newc2

print('new centroid1: ',newc1)
print('new centroid2: ',newc2)

