---
layout: post
title:  "Principal Component Analysis a.k.a. PCA applied to lots of stellar chemical abundance measurements"
date:   2017-04-03
use_math: true
categories: PCA, chemical abundances, dimensionality reduction
---
## Dimensionality reduction: 
### Carbon-enhanced metal poor $\($CEMP$\)$ stars and their chemical abundance patterns

This notebook is a blast from the past--my undergraduate research thesis was written on carbon-enhanced metal-poor stars, and I found some old data in my Gmail account that was great for illustrating the concept of dimensionality reduction. 

First, some background on the science behind what we're doing and why...

### What is a CEMP star and why do some people care about them???

In astronomy, a _metal_ is any element that is heavier than hydrogen or helium. $\($Yes, oxygen counts as a metal. So do carbon, nitrogen, etc.$\)$ So a _metal-poor_ star is one which contains relatively low levels of elements heavier than H/He compared to our own Sun $\($usually at least 10 times lower$\)$. 
![](http://astronomy.swin.edu.au/cms/cpg15x/albums/userpics/PeriodicTable.gif)

Astronomers also tend to measure quantities in logarithmic scales. For example, to indicate that some star has the same ratio of oxygen atoms to hydrogen atoms as our Sun, we would write that this star's oxygen abundance as $[O/H]=0.0$, i.e. the ratio of oxygen to hydrogen in this star is $10^{~0} = 1$ times the oxygen/hydrogen ratio in our Sun. Another more oxygen-poor star might have an abundance of $[O/H]=-2$, meaning that its oxygen/hydrogen ratio is $10^{~-2}=0.01$ times that in Sun.

Carbon-enhanced metal-poor $\($CEMP$\)$ stars are stars with very low iron $\($Fe$\)$ content relative to the Sun, but not-as-low carbon content as you might expect, given how low their other metal abundances are.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from copy import copy 

```

### Preliminaries: the curse of dimensionality

Consider a D-dimensional uniform distribution with coordinates between [r,-r] on each axis. 
What is the probability of randomly drawing a point that lies within a distance $ r=\sqrt{x_1^2+\dots+x_{D}^2} $ of the origin? 

The volume of a hypersphere with radius r is $$ V_D(r)=\frac{2 r^D \pi^{D/2}}{D~\Gamma(D/2)} $$ so this probability is given by $$ f_D=\frac{V_D(r)}{(2r)^D} = \frac{\pi^{D/2}}{D~2^{D-1}~\Gamma(D/2)} $$ The limit of which goes to zero as D approaches infinity: $$  \lim_   {D\to \infty} f_d = 0 $$

The larger the dimensionality of your data set, the more difficult it is to evenly sample its volume. 


### CEMP stars as high-dimensional datasets
As an example below, we have approximately 100 stars $\($data points$\)$ with 23 measured chemical abundances $\($they live in a 23-dimensional parameter space$\)$. We could try and look for correlations between difference elemental abundances by pairing up every possible combination of elements, but that $\(1\)$ that would be a lot of pairings, and $\(2\)$ that wouldn't tell us anything if the structure in the data involved combinations of 3 or more coordinates $\($abundances$\)$. 




```python
#We have ~100 carbon enhanced metal poor stars with 23 chemical abundances measured [Fe/H] and [X/Fe]

#importing and floatify data
infile=open('CEMPabundances.dat','r')
per_row = []
for line in infile:
    per_row.append(line.split('\t'))
elem_abundance_arr_str = list(zip(*per_row))

truncated_elem_abundance_arr=[]
tot_elem_abundance_arr=[]
for arr0 in elem_abundance_arr_str:
    tot_elem_abundance_arr.append([float(i) for i in arr0])

#remove unmeasured elements
for n in range(len(tot_elem_abundance_arr)):
    arr0=tot_elem_abundance_arr[n]
    truncated_elem_abundance_arr.append([x for x in arr0 if x!=-10000.])
    
#read labels for abundances
infile.close()
#for arr in elem_abundance_arr:
#    for i in range(len(arr)):
#        arr[i]=float(arr[i])

infile=open('elem_list.dat','r')
per_row = []
for line in infile:
    per_row.append(line.split('\t'))
elems=per_row[0]
infile.close()

#histogram distributions in abundances
fig = plt.figure(figsize=(20, 20))
for n in range(0,23):
    ax=fig.add_subplot(5,5,(n+1))
    ax.hist(truncated_elem_abundance_arr[n], 10, normed=1, facecolor='blue', alpha=0.4)
    #ax.set_xlabel(elems[n])
    ax.annotate(elems[n] ,xy=(0.05,0.9), xycoords='axes fraction', size=14)

plt.show()
```


![png](/assets/CEMP_stellar_abundances_files/CEMP_stellar_abundances_3_0.png)


You can kind of see that the distributions of chemical abundances for some elements (Europium, or Eu, looks the most obvious) appear to be bimodal. Might there be distinct sub-populations of these stars which can be grouped based on their abundances of a few of the elements above? $\($Spoiler, yes--we can sort of predict which elements have the most variance between populations using stellar evolution theory. But PCA makes this really obvious even if you don't know which evolutionary tracks produce barium versus europium.$\)$

### Searching for structure in a 23-dimensional data set

Without reducing the dimensionality, we can play around and try to search for some structure in the abundances. Looking for correlations between two dimensions is doable. If we were looking for structure in a plane defined by just two of these axes, we could make a triangle plot $\($no room here though$\)$ and visually inspect for any pairs of related abundances. We could do something similar to search for structure in the data that depended on 3 dimensions, but that would be hard to visualize and we'd have $\mathcal{O}(D^3)$ combinations to test, and so on for even more dimensions of the data. 


```python
Dim1=17  #0-22    cheat: try combinations of 1,17,18,20,21,22
Dim2=21 #0-22

Nobjs=len(tot_elem_abundance_arr[0])

#reform this array since next cell is changing it for whatever reason
tot_elem_abundance_arr=[]
for arr0 in elem_abundance_arr_str:
    tot_elem_abundance_arr.append([float(i) for i in arr0])

#only use objects with measurements for both elements
arr1,arr2=[],[]
for obj in range(Nobjs):
    if tot_elem_abundance_arr[Dim1][obj]!=-10000. and tot_elem_abundance_arr[Dim2][obj]!=-10000.:
        arr1.append(tot_elem_abundance_arr[Dim1][obj])
        arr2.append(tot_elem_abundance_arr[Dim2][obj])

fig=plt.figure(figsize=(3, 3))
ax=fig.add_subplot(111)

ax.plot(arr1,arr2, 'o')
ax.annotate('N$_{obj}$='+repr(len(arr1)) ,xy=(0.05,0.9), xycoords='axes fraction', size=14)
ax.set_xlabel(elems[Dim1],fontsize=22)
ax.set_ylabel(elems[Dim2],fontsize=22)
```




    <matplotlib.text.Text at 0x7f6937b85588>




![png](/assets/CEMP_stellar_abundances_files/CEMP_stellar_abundances_6_1.png)


I cheated and used my prior knowledge to pick a combination of two elements that might be expected to have some structure in their 2D scatter plot. Europium $\($Eu$\)$ is produced in what is known as the r $\($rapid$\)$ process, which occurs at the late stages of life for high-mass stars. Therefore we expect a population of stars with higher Eu abundance (those which accreted  material from a dying high-mass binary companion) and a population with lower Eu abundance $\($stars which did not have a high-mass companion to accrete Eu from$\)$. Barium $\($Ba$\)$ is produced in the s $\($slow$\)$ process, which occurs in the late stages of a low-mass star's life. Thus we might expect that stars with higher Ba content might belong to a different group than stars with higher Eu content.

However, this isn't apparent in the figure above. By eye, the two clumps of data points with [Ba/Fe]>1 have a correlation between Ba and Eu abundance. This is because their Ba and Eu content is mostly obtained through *accretion* of material from a companion star, or is already present in the gas that the star formed out of. Therefore a star might have a higher Ba or Eu abundance simply because it formed more recently, out of gas that was enriched by prior generations of stars. To disentangle this effect, we need to also consider the abundances of Fe and C, which give us information about how old a star is and how much material it has accreted from its companion.

### PCA decomposition

The code below does a principal components analysis of the dataset and returns the top three PCA vectors. Not all stars have all 23 elements measured. I fill in the unmeasured value with the median of the rest of the data, which is a simple way to deal with this. There are probably better ways of taking care of this consideration though.


```python
#unmeasured elements replaced with median of measurements
n_components=8
processed_arr=[]    
copy_arr = list(copy(tot_elem_abundance_arr))

# replace unrecorded values with median
for n in range(len(copy_arr)):
    arr=copy_arr[n]
    processed_arr.append(arr)
    temp_arr=[]
    for i in range(Nobjs):
        if arr[i]!=-10000:
            temp_arr.append(arr[i])
    for i in range(Nobjs):
        if arr[i]==-10000:
            processed_arr[n][i]=np.median(temp_arr)
            
# normalize data
copy_arr = list(copy(processed_arr))
std_processed_arr = StandardScaler().fit_transform(np.transpose(copy_arr))


'''
copy_arr = list(copy(processed_arr))
for n in range(len(copy_arr)):
    arr=copy_arr[n]
    max_value = np.max( abs(np.array(copy_arr[n])))
    processed_arr[n] = np.array(processed_arr[n]/max_value)
'''


# tranpose so each row is all chemical abundances for one star  
#std_processed_arr=np.transpose(std_processed_arr)
            
pca = PCA(n_components)

pca.fit(std_processed_arr)

pca_score = pca.explained_variance_ratio_
V = pca.components_
PC1, PC2, PC3=V[0],V[1],V[2]


print( 'Percent of variance in each PC' )
print(pca.explained_variance_ratio_) 
print (repr(np.sum(pca_score))+' percent of variance captured by first '+repr(n_components)+' eigenvectors')


# plot orientations of principal components
fig=plt.figure(figsize=(20, 12))
ax=fig.add_subplot(311)

x=np.linspace(0,22,23)
zeroline=np.zeros(23)
ax.set_xlim([-1,24])
ax.set_ylim([-1,1])
ax.set_ylabel('PC 1',fontsize=20)
ax.plot(x,zeroline,'--',color='grey')

for n in range(len(elems)):
    ax.annotate(elems[n] ,xy=(n,0.6+((-1)**n)*0.05), xycoords='data',ha="center", size=14)
ax.plot(x,PC1,'o')

ax2=fig.add_subplot(312)
ax2.set_xlim([-1,24])
ax2.set_ylim([-1,1.0])
ax2.set_ylabel('PC 2',fontsize=20)
ax2.plot(x,zeroline,'--',color='grey')

for n in range(len(elems)):
    ax2.annotate(elems[n] ,xy=(n,0.7+((-1)**n)*0.05), xycoords='data',ha="center", size=14)
ax2.plot(x,PC2,'o')

ax3=fig.add_subplot(313)
ax3.set_xlim([-1,24])
ax3.set_ylim([-1,1.0])
ax3.set_ylabel('PC 3',fontsize=20)
ax3.plot(x,zeroline,'--',color='grey')

for n in range(len(elems)):
    ax3.annotate(elems[n] ,xy=(n,0.7+((-1)**n)*0.05), xycoords='data',ha="center", size=14)
ax3.plot(x,PC3,'o')

```

    Percent of variance in each PC
    [ 0.2001063   0.12839156  0.09631772  0.0890388   0.06421381  0.0601888
      0.04742005  0.04504102]
    0.73071806833955022 percent of variance captured by first 8 eigenvectors





    [<matplotlib.lines.Line2D at 0x7f6935d7a908>]




![png](/assets/CEMP_stellar_abundances_files/CEMP_stellar_abundances_9_2.png)


The first principal component PC1 accounts for about 20% of the variance. The elements with the highest contribution to PC1--other than Fe and C--are heavy elements that are formed through either the r- or s- processes. This isn't surprising if you are familiar with the different tracks of stellar evolution--the mass of the star will determine whether it undergoes r- or s- process synthesis at the end of its life. 

The second principal component PC2 accounts for about 13% of the variance. Aside from Fe, the main components of PC2 are lighter elements: N, Na, Mg, Al, Si, and Ca. Mg, Si, and Ca are so-called 'alpha elements', which are produced in type II supernovae. I'm not exactly sure how to interpret or explain the composition of PC2.

### Plot in PCA space and compare to existing classification

Finally, I take the original data and look at its 2D projection onto planes defined by different PCA eigenvectors. Each of these stars has been classified into a different subclass of CEMP stars in previous studies. At zeroth order, astronomers look at [Ba/Fe], [Eu/Fe], and [La/Fe] and make cuts in those abundances to define what group a star belongs in. For example, any star with [Ba/Fe] > $x_1$ and [Eu/Fe]<$x_2$ gets classified as type 's', and so on. In the future I'd be interested in running some clustering algorithm on the PCA projection and seeing if I can recover these classifications.


```python

#import type of each star so they can be labeled by color
infile=open('type_list.dat','r')
cemp_type = []
for line in infile:
    cemp_type.append(line.split()[0])
color_list = []
for t, starType in enumerate(cemp_type):
    if starType =='s':
        color_list.append('salmon')
    elif starType =='no':
        color_list.append('purple')
    elif starType =='rs':
        color_list.append('gold')
    else:
        color_list.append('turquoise')
        

# Project data in PCA space
pca_projection = np.transpose(pca.fit_transform(std_processed_arr))

# Pick two PCA eigenvectors (1-8)
pca_axes = (0,1)
xAxis = pca_projection[pca_axes[0]]
yAxis = pca_projection[pca_axes[1]]

# Plot 2D projection in PCA_x, PCA_y
fig=plt.figure(figsize=(12,12))

ax=fig.add_subplot(223)

# Pick two PCA eigenvectors (1-8)
pca_axes = (0,1)
xAxis = pca_projection[pca_axes[0]]
yAxis = pca_projection[pca_axes[1]]

ax=fig.add_subplot(223)
ax.scatter(xAxis,yAxis, color=color_list, picker=True)
ax.set_xlabel('PCA '+ repr(pca_axes[0]+1))
ax.set_ylabel('PCA '+ repr(pca_axes[1]+1))
ax.set_xlim([np.min(xAxis)*1.2, np.max(xAxis*1.2)])
ax.set_ylim(np.min(yAxis)*1.2, np.max(yAxis*1.2))


ax=fig.add_subplot(221)
pca_axes = (0,2)
xAxis = pca_projection[pca_axes[0]]
yAxis = pca_projection[pca_axes[1]]

ax.scatter(xAxis,yAxis, color=color_list, picker=True)
ax.set_xlabel('PCA '+ repr(pca_axes[0]+1))
ax.set_ylabel('PCA '+ repr(pca_axes[1]+1))
ax.set_xlim([np.min(xAxis)*1.2, np.max(xAxis*1.2)])
ax.set_ylim(np.min(yAxis)*1.2, np.max(yAxis*1.2))

ax=fig.add_subplot(224)
pca_axes = (2,1)
xAxis = pca_projection[pca_axes[0]]
yAxis = pca_projection[pca_axes[1]]

ax.scatter(xAxis,yAxis, color=color_list, picker=True)
ax.set_xlabel('PCA '+ repr(pca_axes[0]+1))
ax.set_ylabel('PCA '+ repr(pca_axes[1]+1))
ax.set_xlim([np.min(xAxis)*1.2, np.max(xAxis*1.2)])
ax.set_ylim(np.min(yAxis)*1.2, np.max(yAxis*1.2))

ax.plot([-99],[-99], 'o', label='s', color='salmon')
ax.plot([-99],[-99], 'o', label='rs', color='gold')
ax.plot([-99],[-99], 'o', label='low-s', color='turquoise')
ax.plot([-99],[-99], 'o', label='no', color='purple')

ax.legend(bbox_to_anchor=(0.7, 2.05), fontsize=20)
```




    <matplotlib.legend.Legend at 0x7f6934702240>




![png](/assets/CEMP_stellar_abundances_files/CEMP_stellar_abundances_12_1.png)

