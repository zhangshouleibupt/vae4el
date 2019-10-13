from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np

class ELDataset(Dataset):
    def __init__(self,vocab,filepath):
        self.vocab=vocab

        self.pairs=[]

        with open(filepath, "r") as f:
            for line in f:
                elements = line.split("<--->")
                self.pairs.append((elements[1]+" "+elements[3],elements[4]+" "+elements[5],elements[-1]))


    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        #index

        mention_index = [self.vocab.get_word_index(word) for word in self.pairs[idx][0].split(" ")]
        entity_index = [self.vocab.get_word_index(word) for word in self.pairs[idx][1].split(" ")]

        #pad
        maxlen=64
        if len(mention_index)<maxlen:

            mention_index.extend([0]* (maxlen-len(mention_index)))

        else:

            mention_index = mention_index[:maxlen]

        if len(entity_index) < maxlen:

            entity_index.extend([0]* (maxlen-len(entity_index)))
        else:
            entity_index = entity_index[:maxlen]


        mention=torch.tensor(mention_index)
        entity = torch.tensor(entity_index)
        label=int(self.pairs[idx][2])

        return (mention,entity,label)


