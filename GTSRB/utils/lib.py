
from __future__ import print_function
import numpy as np

def softmax(x):
	x = x - np.max(x)
	x = np.exp(x)/np.sum(np.exp(x))
	return x
