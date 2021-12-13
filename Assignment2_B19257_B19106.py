# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# All required libraries are imported here
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt 
import cv2 # library for computer vision
from sklearn.metrics import confusion_matrix, accuracy_score

plt.rcParams["figure.figsize"] = (13,10)
plt.style.use("ggplot")

#----------------------------------------Question 1----------------------------------------#

# function that gets the whole dataframe from the file passed as argument
def dataf(path,sep):
    
    data = pd.DataFrame()
    for i in range(len(path)):
        X = pd.read_csv(path[i], sep=sep, header=None)  
        X[len(X.columns)] = i
        data = pd.concat([data,X],ignore_index=True)
    data.columns = ['X','Y','label']
    return data 

# Function for calculating the euclidean distaance between two points.
def euclidean_distance(row1, row2):

    t = 0
    for i in  range(len(row1)):
        t+=(row1[i]-row2[i])**2  
    return math.sqrt(t)


# This function finds the distortion from k-mean
def distortion(df,mean):
    
    grouped = df.groupby('label')
    distortion = 0
    means = []
    for i,j in grouped:
        means.append(j.iloc[:,:-1].mean().tolist())
        for k in j.index:
            distortion += euclidean_distance(j.loc[k][:-1].tolist(),mean.iloc[0])
    return distortion, pd.DataFrame(means)

# Calculation of the k-mean
def Calculate_k_mean(df,k):
    
    set_random_mean = df.sample(n=k)
    dist = 0

    while(True):
        label=[]
        df.drop('label',inplace=True, axis=1)
        for i in df.index:
            euc_dist = [] 

            for j in range(k):

                
                euc_dist.append(euclidean_distance(df.iloc[i].tolist(),set_random_mean.iloc[j].tolist()))
            label.append(np.argmin(euc_dist))
        df["label"] = label
        new_dist, new_mean = distortion(df,set_random_mean)
        if(abs(new_dist-dist)<0.1):
            break
        dist = new_dist
        set_random_mean = new_mean
    return df, set_random_mean

# Scatter Plot 
def scatter_plot(train):
    
    gkk = train.groupby('label')
    colors = ['red','blue','gold','green','#ac1bc1','#d11554','lightskyblue']
    for i,j in gkk:
        plt.scatter(j['X'],j['Y'],c=colors[i],label="Class "+str(i+1),alpha=0.8)
    plt.title("Scatter Plot for NLS data using K-mean Clustering",fontsize=20)
    plt.xlabel("Attribute - 1",fontsize=14)
    plt.ylabel("Attribute - 2",fontsize=14)
    plt.legend()
    plt.show()     


def Q1():
    print("#"+"-"*40+ " Question 1 "+"-"*40+"#")
    print("\nK-means clustering for the NLS data")
    df = dataf(['./nls_data/class1.txt','./nls_data/class2.txt'],",")
    lbl = df['label']
    data, means = Calculate_k_mean(df,2)
    conf_mat = confusion_matrix(lbl,data['label']) # finding confusion matrix
    accu = 100 * accuracy_score(lbl,data['label'])  # finding accuracy of K-mean clustering
    print("\n* Confusion Matrix:\n",conf_mat)
    print("* Accuracy: %5.4f" % (accu))
    scatter_plot(data)

#----------------------------------------Question 2----------------------------------------#


# Function for segmenting image using k-mean.   
def segment_image(data,k, img, t):
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center) 
    res = center[label.flatten()]
    if t == 1:
        ne = []
        for i in res:
            ne.append(i[:-2])
        res = np.array(ne)   
    return res.reshape((img.shape))    

# Here for different values of k we have plotted the rendered image
def plot_image(data, img, t):
    
    l = [2,5,10,20]
  
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    for i in range(len(l)):
        plt.subplot(1,len(l),i+1),plt.imshow(segment_image(data,l[i],img,t))
        plt.title('K = %i' % l[i]), 
        plt.xticks([]), plt.yticks([])
    plt.show()


def Q2():
    print("#"+"-"*40+ " Question 2 "+"-"*40+"#")
    #rendering the given image into a 2D vector to be used for segmentation
    image = cv2.imread('Image.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vect = img.reshape((-1,3))
    vect = np.float32(vect)
    
    # This is the original image
    plt.rcParams["figure.figsize"] = (10,5)
    plt.title("Original Image",fontsize=20)
    plt.imshow(img)
    plt.show()
    
    print("K-means clustering-based segmentation of the given image.")
    print("i). When using only pixel colour values as features")
    
    plot_image(vect, img,0)
    
    pixel_width, pixel_height, dim = img.shape
    count = 0
    array1 = []
    for i in range(pixel_width):
        for j in range(pixel_height):
            col1, col2, col3 = vect[count]
            array1.append([int(col1), int(col2), int(col3), int(i)*255/pixel_width, int(j)*255/pixel_height])
            count+=1
    vect = np.array(array1)    
    vect = np.float32(vect)
    print("ii). When using both pixel colour and location values as features")
    plot_image(vect, img,1)

  
#Q1()
Q2()












