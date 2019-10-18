from collections import defaultdict

def build_voc(file_path):
    word2freq = defaultdict(int)
    idx = 0
    with open(file_path,'r',encoding='utf8') as f:
        for line in f:
            items = line.rstrip('\n').split('<--->')
            strs = (items[1],items[3],items[4],items[5])
            for s in strs:
                words = s.split(' ')
                for word in words:
                    word2freq[word] += 1
    with open('../data/voc.dict','w',encoding='utf8') as f:
        for word,freq in word2freq.items():
            f.write('%s %d\n'%(word,freq))

if __name__  == "__main__":
    build_voc('../data/aida.data')