import pandas as pd
import numpy as np

df = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
df.head(n=10)


data = df.iloc[:,-3:].values

#1- & 2- dim shape size and data type
print(f"dim = {data.ndim }, shape = {data.shape}, size = {data.size}")
print(f"Data type = {data.dtype}")

# 3- target
print(data[:,2])
print(f"target shape  {data[:,2].shape}")
print(data[:,2].reshape(1*-1))

#4- data 3 observation
print(data[::3,:])

#5- concatination first and last col
firstFeature = data[:,0:1]
LastFeature  = data[:,2:]
print(np.hstack((firstFeature,LastFeature)))

#6- reverse rows
print(data[::-1])

#7- max and min hight
heights = data[:,:1]
print(f"max heights = {np.max(heights)}")
print(f"min heights = {np.min(heights)}")

#8- max and min wieghts
wieghts = data[:,1:2]
print(f"max wieghts = {np.max(wieghts)}")
print(f"min wieghts = {np.min(wieghts)}")

#9- mean and standard deviation of hight 
heights = data[:,:1]
print(f"mean heights = {np.mean(heights):.2f}")
print(f"std heights = {np.std(heights):.2f}")

#10- mean and standard deviation of wieghts 
wieghts = data[:,1:2]
print(f"mean wieghtss = {np.mean(wieghts):.2f}")
print(f"std wieghtss = {np.std(wieghts):.2f}")

#11- 25th,27th percentile and the median for hight
heights = data[:,:1]
print(f"median heights = {np.median(heights):.2f}")
print(f"percentile of 25 heights = {np.percentile(heights,25):.2f}")
print(f"percentile of 75 heights = {np.percentile(heights,75):.2f}")

#12- 25th,27th percentile and the median for wieghts
wieghts = data[:,1:2]
print(f"median wieghts = {np.median(wieghts):.2f}")
print(f"percentile of 25 wieghts = {np.percentile(wieghts,25):.2f}")
print(f"percentile of 75 wieghts = {np.percentile(wieghts,75):.2f}")

#13- normalize of hight and wieghts
print(f"normalize of hight = {(heights-np.mean(heights))/np.std(heights)}")
print(f"normalize of wieghts= {(wieghts-np.mean(wieghts))/np.std(heights)}")

#14- min max scaler
print(f"min max scaler of hight = {((heights-np.min(heights))/(np.max(heights)-np.min(heights)))}")
print(f"min max scaler of wieghts = {((wieghts-np.min(wieghts))/(np.max(wieghts)-np.min(wieghts)))}")

#15- indices of the tallest people
print(f"indices of the tallest people = {np.argmax(heights)}")

#16- hight and wight square
print(f"square heights = {np.square(heights)}")
print(f"square wieghts = {np.square(wieghts)}")

#17- hight and wight sqrt
print(f"sqrt heights = {np.sqrt(heights)}")
print(f"sqrt wieghts = {np.sqrt(wieghts)}")
