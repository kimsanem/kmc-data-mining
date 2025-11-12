import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
# from normalize import min_max, z_score, decimal_normalize

def min_max(number, min_range=0, max_range=1):
    ''' desc: this function is used for return value of number after normalization of data using min-max technique
        function_name: min_max
        no. of argument: 3
        argument type: list,num,num
    '''

    min_no = min(number)
    max_no = max(number)

    return (min_range + ((number-min_no)*(max_range-min_range)) / (max_no-min_no))

def z_score(number):
    ''' desc: this function is used for return value of number after standardization of data using z-score technique 
        function_name: z_score
        no. of argument: 1
        argument type: number
    '''
    mean = np.mean(number)
    std_dev = np.std(number)
    
    return (number-mean)/std_dev

# number = [115,120,250,85,700,122]

# print("Decimal Normalization: ", decimal_normalize(number))
# print("Min-Max Normalization: ", min_max(number))
# print("Z-Score Standardization: ", z_score(number))


df = sns.load_dataset('iris')
print(df)
# sepal_length = df['sepal_length']
sepal_length = df['sepal_length'].to_numpy().reshape(-1, 1)
range = np.arange(len(sepal_length))


fix, axes = plt.subplots(2,3, figsize=(15,8))


# Original Data
axes[0,0].scatter(df['sepal_length'], range, color = 'yellow', alpha = 0.6)
# axes[0,0].boxplot(df['sepal_length'], patch_artist=True, boxprops=dict(facecolor='yellow', alpha=0.6))
axes[0,0].set_title('Before Min-Max Normalization')
axes[0,0].set_xlabel('Range')
axes[0,0].set_ylabel('Sepal Length(cm)')


# Custom Min-Max
new_sepal_length1 = min_max(df.sepal_length)
axes[0,1].scatter(new_sepal_length1, range, color = 'blue', alpha = 0.6)
# axes[0,1].boxplot(new_sepal_length, patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.6))
axes[0,1].set_title('After Min-Max Normalization')
axes[0,1].set_xlabel('Range [0,1]')
axes[0,1].set_ylabel('Sepal Length(cm)')

# Custom Z-Score
new_sepal_length2 = z_score(df.sepal_length)
axes[0,2].scatter(new_sepal_length2, range, c="green", alpha=0.6)
axes[0,2].set_title("After Z-Score Normalization")
axes[0,2].set_xlabel("Range")
axes[0,2].set_ylabel("Sepal Length(cm)")

# sklearn Min-Max
minmax_scaler = MinMaxScaler()
sepal_minmax = minmax_scaler.fit_transform(sepal_length)
axes[1,0].scatter(sepal_minmax, range, color="red", alpha=0.6)
axes[1,0].set_title("sklearn Min-Max Normalization")
axes[1,0].set_xlabel("Range")
axes[1,0].set_ylabel("Sepal Length(cm)")

#sklearn Z-Score
zscore_scaler = StandardScaler()
sepal_zscore = zscore_scaler.fit_transform(sepal_length)
axes[1,1].scatter(sepal_zscore, range, color="pink", alpha=0.6)
axes[1,1].set_title("sklearn Z-Score Normalization")
axes[1,1].set_xlabel("Range")
axes[1,1].set_ylabel("Sepal Length(cm)")

plt.tight_layout()
plt.show()