import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
import math
# import modules
import numpy as np
from scipy import stats
# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt
# evaluate a lda model on the dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('non_parametric_data.csv')
y=df['label']
x1 = df['feature1']
x2 = df['feature2']
x3 = df['feature3']
x4 = df['feature4']
x5 = df['feature5']
x6 = df['feature6']
x7 = df['feature7']
x8 = df['feature8']
x9 = df['feature9']
x10 = df['featuture10']



original_data1=[]
original_y1=[]
for i in range (len(x1)):
    if x1[i]>0:
        original_data1.append(x1[i])
        original_y1.append(y[i])

original_data2=[]
original_y2=[]
for i in range (len(x2)):
    if x2[i]>0:
        original_data2.append(x2[i])
        original_y2.append(y[i])


original_data3=[]
original_y3=[]
for i in range (len(x3)):
    if x3[i]>0:
        original_data3.append(x3[i])
        original_y3.append(y[i])


original_data4=[]
original_y4=[]
for i in range (len(x4)):
    if x4[i]>0:
        original_data4.append(x4[i])
        original_y4.append(y[i])


original_data5=[]
original_y5=[]
for i in range (len(x5)):
    if x5[i]>0:
        original_data5.append(x5[i])
        original_y5.append(y[i])



original_data6=[]
original_y6=[]
for i in range (len(x6)):
    if x6[i]>0:
        original_data6.append(x6[i])
        original_y6.append(y[i])



original_data7=[]
original_y7=[]
for i in range (len(x7)):
    if x7[i]>0:
        original_data7.append(x7[i])
        original_y7.append(y[i])



original_data8=[]
original_y8=[]
for i in range (len(x8)):
    if x8[i]>0:
        original_data8.append(x8[i])
        original_y8.append(y[i])



original_data9=[]
original_y9=[]
for i in range (len(x9)):
    if x9[i]>0:
        original_data9.append(x9[i])
        original_y9.append(y[i])



original_data10=[]
original_y10=[]
for i in range (len(x10)):
    if x10[i]>0:
        original_data10.append(x10[i])
        original_y10.append(y[i])




# transform training data & save lambda value
fitted_data1, fitted_lambda1 = stats.boxcox(original_data1)
fitted_data2, fitted_lambda2 = stats.boxcox(original_data2)
fitted_data3, fitted_lambda3 = stats.boxcox(original_data3)
fitted_data4, fitted_lambda4 = stats.boxcox(original_data4)
fitted_data5, fitted_lambda5 = stats.boxcox(original_data5)
fitted_data6, fitted_lambda6 = stats.boxcox(original_data6)
fitted_data7, fitted_lambda7 = stats.boxcox(original_data7)
fitted_data8, fitted_lambda8 = stats.boxcox(original_data8)
fitted_data9, fitted_lambda9 = stats.boxcox(original_data9)
fitted_data10, fitted_lambda10 = stats.boxcox(original_data10)



for i in range(len(fitted_data1)):
    with open('data1.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y1[i],fitted_data1[i]])
        f.close()
for i in range(len(fitted_data2)):
    with open('data2.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y2[i],fitted_data2[i]])
        f.close()
for i in range(len(fitted_data3)):
    with open('data3.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y3[i],fitted_data3[i]])
        f.close()
for i in range(len(fitted_data4)):
    with open('data4.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y4[i],fitted_data4[i]])
        f.close()
for i in range(len(fitted_data5)):
    with open('data5.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y5[i],fitted_data5[i]])
        f.close()
for i in range(len(fitted_data6)):
    with open('data6.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y6[i],fitted_data6[i]])
        f.close()
for i in range(len(fitted_data7)):
    with open('data7.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y7[i],fitted_data7[i]])
        f.close()
for i in range(len(fitted_data8)):
    with open('data8.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y8[i],fitted_data8[i]])
        f.close()
for i in range(len(fitted_data9)):
    with open('data9.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y9[i],fitted_data9[i]])
        f.close()
for i in range(len(fitted_data10)):
    with open('data10.csv','a', newline='') as f:
        writer=csv.writer(f)
        writer.writerow([original_y10[i],fitted_data10[i]])
        f.close()
