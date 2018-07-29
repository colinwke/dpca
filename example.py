"""example of density peak cluster algorithm."""
import pandas as pd

from dpca import DensityPeakCluster

# file name
file = "spiral"

# load data
data = pd.read_csv(r"data/data/%s.txt" % file, sep="\t", header=None)

# dpca model
# plot decision graph to set params `density_threshold`, `distance_threshold`.
dpca = DensityPeakCluster(density_threshold=8, distance_threshold=5, anormal=False)

# fit model
dpca.fit(data.iloc[:, [0, 1]])

# print predict label
print(dpca.labels_)

# plot cluster
dpca.plot("all", title=file, save_path="data/result")
