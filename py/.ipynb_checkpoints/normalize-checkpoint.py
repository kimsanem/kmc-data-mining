''' data normalization for standardization are both data preprocessing techniques used to scale and transform data,'''


import numpy as np

def decimal_normalize(number):
    data = np.array(number, dtype=float)
    j = int(np.ceil(np.log10(np.max(np.abs(number)))))
    return data / (10 ** j)

def min_max(number, min_range=0, max_range=1):
    ''' desc: this function is used for return value of number after normalization of data using min-max technique
        function_name: min_max
        no. of argument: 3
        argument type: list,num,num
    '''
    data = np.array(number, dtype=float)
    min_no = min(data)
    max_no = max(data)

    return (min_range + ((data-min_no)*(max_range-min_range)) / (max_no-min_no))

def z_score(number):
    ''' desc: this function is used for return value of number after standardization of data using z-score technique 
        function_name: z_score
        no. of argument: 1
        argument type: list
    '''
    data = np.array(number, dtype=float)
    mean = np.mean(data)
    std_dev = np.std(data)
    
    return (data-mean)/std_dev


number = [115,120,250,85,700,122]
print("Number: ", number)
print("Decimal Normalization: ", decimal_normalize(number))
print("Min-Max Normalization: ", min_max(number))
print("Z-Score Standardization: ", z_score(number))