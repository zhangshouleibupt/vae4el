import pickle

class Vocab():

    def __init__(self):
        self.word2index={}
        self.index2word={}
        self.word_freq={}


        self.add_word("pad")#必须第一个插入
        self.add_word("unk")

    def __len__(self):
        return len(self.word2index)

    def add_word(self,word):
        id=len(self.word2index)
        self.word2index[word]=id
        self.index2word[id]=word
        self.word_freq[word]=1


    def add_string(self,str):
        words=str.split(' ')

        for word in words:
            if word in self.word2index:
                self.word_freq[word]+=1
            else:
                self.add_word(word)


    def read_file(self, filepath):
        with open(filepath, "r") as f:
            for line in f:
                elements = line.split("<--->")
                self.add_string(elements[1])
                self.add_string(elements[4])
                self.add_string(elements[3])
                self.add_string(elements[5])

    def get_word_index(self,word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["unk"]


    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


