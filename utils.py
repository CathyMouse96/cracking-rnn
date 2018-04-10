import os
import numpy as np
from six.moves import cPickle as pickle

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, isTraining=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        vocab_file = os.path.join(self.data_dir, 'vocab.pkl')
        train_file = os.path.join(self.data_dir, 'train.txt')
        valid_file = os.path.join(self.data_dir, 'valid.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')

        assert os.path.exists(self.data_dir), "data directory does not exist"
        assert os.path.exists(vocab_file), "vocab file does not exist"
        assert os.path.exists(train_file), "train file does not exist"
        assert os.path.exists(valid_file), "validation file does not exist"
        assert os.path.exists(test_file), "test file does not exist"

        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.vocab_size = len(self.vocab)
        # print self.vocab

        if isTraining:
            train_data_file = os.path.join(self.data_dir, 'train.npy')
            train_len_file = os.path.join(self.data_dir, 'train_len.npy')
            valid_data_file = os.path.join(self.data_dir, 'valid.npy')
            valid_len_file = os.path.join(self.data_dir, 'valid_len.npy')
            
            if not (os.path.exists(train_data_file) and os.path.exists(valid_data_file)):
                print("reading text file")
                self.train_data, self.train_len, self.valid_data, self.valid_len = \
                self.preprocess(train_file, valid_file, \
                train_data_file, train_len_file, valid_data_file, valid_len_file)
            else:
                print("loading preprocessed files")
                self.train_data = np.load(train_data_file)
                self.train_len = np.load(train_len_file)
                self.valid_data = np.load(valid_data_file)
                self.valid_len = np.load(valid_len_file)
                
            self.num_batches = int(len(self.train_data) / self.batch_size)
            print("number of batches: {}".format(self.num_batches))
            
            self.create_batches()
            self.reset_batch_pointer()
        
        else:
            first_char_probs_file = os.path.join(self.data_dir, 'first_char_probs.pkl')

            if not (os.path.exists(first_char_probs_file)):
                print("generating first char probabilities")
                self.first_char_probs = self.generate_probs( \
                train_file, valid_file, first_char_probs_file)
            else:
                print("retrieving stored probabilities")
                with open(first_char_probs_file, 'rb') as f:
                    self.first_char_probs = pickle.load(f)

    def preprocess(self, train_file, valid_file, \
        train_data_file, train_len_file, valid_data_file, valid_len_file):
        train_data = []
        train_len = []
        with open(train_file, 'r') as f:
            for line in f:
                train_len.append(len(line))
                line = np.array(map(self.vocab.get, line))
                line = np.pad(line, (0, self.seq_length - len(line)), 'constant')
                train_data.append(line)
        train_data = np.array(train_data)
        train_len = np.array(train_len)
        np.save(train_data_file, train_data)
        np.save(train_len_file, train_len)
        print("stored train data")

        valid_data = []
        valid_len = []
        with open(valid_file, 'r') as f:
            for line in f:
                valid_len.append(len(line))
                line = np.array(map(self.vocab.get, line))
                line = np.pad(line, (0, self.seq_length - len(line)), 'constant')
                valid_data.append(line)
        valid_data = np.array(valid_data)
        valid_len = np.array(valid_len)
        np.save(valid_data_file, valid_data)
        np.save(valid_len_file, valid_len)
        print("stored valid data")
        return train_data, train_len, valid_data, valid_len
    
    def create_batches(self):
        x_batches = self.train_data[:self.num_batches * self.batch_size, :]
        x_batches = np.array(np.split(x_batches, self.num_batches, axis=0))
        y_batches = np.copy(x_batches)
        y_batches[:, :, : -1] = x_batches[:, :, 1 : ]
        y_batches[:, :, -1] = x_batches[:, :, 0]
        length_batches = self.train_len[:self.num_batches * self.batch_size]
        length_batches = np.array(np.split(length_batches, self.num_batches))
        self.x_batches = x_batches
        self.y_batches = y_batches
        self.length_batches = length_batches
    
    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        length = self.length_batches[self.pointer]
        self.pointer += 1
        return x, y, length
    
    def reset_batch_pointer(self):
        self.pointer = 0

    def generate_probs(self, train_file, valid_file, first_char_probs_file):
        first_char_probs = {}
        total_cnt = 0
        with open(train_file, 'r') as f:
            for line in f:
                total_cnt += 1
                first_char = line[0]
                if first_char_probs.has_key(first_char):
                    first_char_probs[first_char] += 1
                else:
                    first_char_probs[first_char] = 1
        with open(valid_file, 'r') as f:
            for line in f:
                total_cnt += 1
                first_char = line[0]
                if first_char_probs.has_key(first_char):
                    first_char_probs[first_char] += 1
                else:
                    first_char_probs[first_char] = 1
        for k in first_char_probs.keys():
            first_char_probs[k] = float(first_char_probs[k]) / float(total_cnt)
        with open(first_char_probs_file, 'wb') as f:
            pickle.dump(first_char_probs, f)
