# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:39:37 2022

@author: finkt
"""
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from viral_centrality import viral_centrality

tol = 0.001

f = open("datasets/congress_network/congress_network_data.json")
data = json.load(f)

inList = data[0]["inList"]
inWeight = data[0]["inWeight"]
outList = data[0]["outList"]
outWeight = data[0]["outWeight"]
usernameList = data[0]["usernameList"]

num_activated = viral_centrality(inList, inWeight, outList, Niter=-1, tol=tol)

plt.scatter(
    np.array(range(len(num_activated))),
    num_activated,
    color="red",
    label="Viral Centrality",
)
plt.xlabel("Node ID", fontsize=15)
plt.ylabel("Avg Number Activaated", fontsize=15)

plt.show()
