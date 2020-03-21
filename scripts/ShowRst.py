import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math

dat=pd.read_csv("/home/llg/workspace/P03-HybridDA/Result/KnnPlane.csv")
print(dat.values)
plt.plot(dat.values,'+')
plt.show()
