import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

dat=pd.read_csv("/home/llg/workspace/P03-HybridDA/Result/KnnPlane.csv")
df_array=dat.values
df_list=df_array.tolist()
st=np.zeros((1,30))


K=30
for i in range(K):
   st[0][i]=df_list.count(np.array([i]))

print(st[0])
plt.plot(st[0])
plt.show()
