from torch.utils.data import Dataset
from typing import Any

"""
An abstract class representing a Dataset.

All datasets that represent a map from keys to data samples should subclass it. All subclasses should overwrite __getitem__, supporting fetching a data sample for a given key. Subclasses could also optionally overwrite __len__, which is expected to return the size of the dataset by many ~torch.utils.data.Sampler implementations and the default options of ~torch.utils.data.DataLoader. Subclasses could also optionally implement __getitems__, for speedup batched samples loading. This method accepts list of indices of samples of batch and returns list of samples.
"""

class BabyData(Dataset):
    def __init__(self,data,labels) -> None:
        self.data = data
        self.labels = labels

    def __getitem__(self, index) -> Any:
        return (self.data[index],self.labels[index])
    
    def __len__(self) -> int:
        return len(self.labels)