import numpy as np

data1 = [1,2,3,4,5,6,7,8,9,10]
data2 = [0,2,4,6,8,10,12,14,16,18]

check1, check2, check3 = np.intersect1d(data1, data2, return_indices=True)
print(check1)   
print(check2)
print(check3)

#first output is the common element
#second output is the indices of the common elements in the first array
#third output is the indices of the common elements in the second array