import numpy as np

class DataSet(object) :
    def __init__(self, datapath, captions, vocab_size, EOS) :
        self._feat = []
        self._label = []
        self._cap = []
        self._max_len = 0
        
        for index, name in enumerate(captions) :
            x = np.load(datapath + name['id'] + '.npy')
            self._feat.append(x)

            for sent in name['caption'] :
                self._label.append(index)
                sent.append(EOS)
                self._cap.append(sent) 
                if len(sent) > self._max_len :
                    self._max_len = len(sent)

        self._feat = np.array(self._feat)
        self._label = np.array(self._label)
        self._cap = np.array(self._cap)
        self._datalen = len(self._cap)
        self._feat_timestep = len(self._feat[0])
        self._feat_dim = len(self._feat[0][0])
        self._vocab_size = vocab_size
        self._index_epoch = 0
        self._num_epoch = 0

        return


    def next_batch(self, batch_size = 1) :
        
        x = []
        y = []

        for _ in range(batch_size) :

            if self._index_in_epoch >= self._datalen : 
                random_index = np.arange(0, self.datalen)
                np.random.shuffle(random_index)
                
                self._label = self._label[random_index]
                self._capt = self._capt[random_index]
                
                self._index_epoch = 0
                self._num_epoch += 1

            x.append(self._feat[self._label[self._index_epoch]])
            y.append(self._caption[self._index_epoch])

            self._index_epoch += 1

        return np.array(x), np.array(y)


    def feat(self) :
        return self._feat

    def label(self) :
        return self._label

    def cap(self) :
        return self._cap

    def max_len(self) :
        return self._max_len
    
    def datalen(self) :
        return self._datalen

    def feat_timestep(self) :
        return self._feat_timestep

    def feat_dim(self) :
        return self._feat_dim

    def vocab_size(self) :
        return self._vocab_size

    def index_in_epoch(self) :
        return self._index_epoch

    def num_epoch(self) :
        return self._num_epoch