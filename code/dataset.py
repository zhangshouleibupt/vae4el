from torch.utils.data import Dataset
import torch
class ELDataset(Dataset):
    def __init__(self,el_dict,filepath):
        self.dictionary = el_dict
        self.pairs=[]
        self.max_len = 64
        with open(filepath, "r") as f:
            for line in f:
                elements = line.split("<--->")
                self.pairs.append((elements[1]+" "+elements[3],elements[4]+" "+elements[5],elements[-1]))
                
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mention_index = [self.dictionary.index(word) for word in self.pairs[idx][0].split(" ")]
        entity_index = [self.dictionary.index(word) for word in self.pairs[idx][1].split(" ")]
        if len(mention_index) < self.max_len:
            mention_index.extend([self.dictionary.pad()]* (self.max_len-len(mention_index)))
        else:
            mention_index = mention_index[:self.max_len]
        if len(entity_index) < self.max_len:
            entity_index.extend([self.dictionary.pad()]* (self.max_len-len(entity_index)))
        else:
            entity_index = entity_index[:self.max_len]
        mention=torch.tensor(mention_index)
        entity = torch.tensor(entity_index)
        label=int(self.pairs[idx][2])
        return (mention,entity,label)


