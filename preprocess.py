import os
#from torch_em.model import UNETR
from torch.utils.data import DataLoader
#from torch.optim import Adam
import torch
from unetrdata import CentroidVectorData

ds = CentroidVectorData("dfki/train")
testds = CentroidVectorData("dfki/test")

print("Preprocessing done, all burst should be cached now")
