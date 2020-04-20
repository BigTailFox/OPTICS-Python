# OPTICS algorithm
## Introduction
An implement of OPTICS(Ordering Points To Identify The Clustering Structure) algorithm in python
## Quick Start
```
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from optics import OPTICS
```
```OPTICS``` class need a pandas ```DataFrame``` to construct, the by-passed ```DataFrame```'s labels colum should have been trimmed off. The native distance calculating method was only designed for contineous attributes. If need to handle nominal or categorical attributes, re-write the distance calculating function.
```
df = pd.read_table("dataset.txt")
df1 = df.iloc[:,:-1]
model = OPTICS(df1)
```
Run the ordering method, this could take some minutes when deal with a large amount of data. 2 Arguments need to be passed in, the ```Eps``` for max epsilon and ```MinPts``` for min points number. Usually, ```Eps``` could be set with ```inf``` and ```MinPts``` with the dimensions of data entries + 1.

Notice: the ```MinPts``` does NOT contain the core point itself
```
model.optics(Eps=float('inf'), MinPts=4)
```
Now the ordering result should be written to ```model.result_queue```, e.g. ```model.result_queue[0]``` is 17 means data entry ```df1.iloc[17,:]``` is the first element of ordering result queue.

And in ```model.core_distances``` and ```model.reachabel_distances``` are the cd and rd of every point accroding to the order of dataframe (not result queue).

Accroding to spirit of OPTICS, now an ordered reachable distance graph should be painted to see which ```Eps``` we should choose to get the best clustering result. This module dosen't provide visualization method, a 3rd-party plot lib is needed such as ```matplotlib```.

Once you decide which ```Eps``` to use, you can extract the final clustering result into ```model.cluster_labels```, it's ordered by the dataframe instead of ```model.result_queue```.
```
model.cluster_extract(Eps=4)
```
## Dependencies
1. python after version 3.5
2. numpy
3. pandas
4. matplotlib