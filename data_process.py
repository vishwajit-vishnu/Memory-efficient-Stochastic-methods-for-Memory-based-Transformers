
#data_process.py
# This is not complete for lm1b dataset

import os,sys;
import torch;
from collections import Counter;

class Vocab(object):
    def __init__(self, dataset, dir_path, sort_vocab):
        assert os.path.exists(dir_path) , 'No such file found for creating vocab';

        self.dataset= dataset ;
        self.sort_vocab= sort_vocab;
        # The symbols(sym) could be a word or a character
        self.sym2idx= {}
        self.sym2count= {} ;
        self.idx2sym= [];

        file_paths= self._get_file_paths(dir_path);
        for file_path in file_paths:
            assert os.path.exists(dir_path) , 'No such file found:{}'.format(file_path) ;
            with open(file_path, 'r', encoding="utf-8") as f:
                for line in f:
                    line= line.strip();
                    if(dataset in ['ptb']):
                        line=line.lower();
                    words = line.split();
                    words= _add_eos(dataset, words);     #New LINE ADDED
                    for sym in words:
                        if(self.sort_vocab):
                            self.sym2count[sym]= self.sym2count.get(sym,0) +1;
                        elif (sym not in self.sym2idx):
                            self.sym2idx[sym]= len(self.idx2sym);
                            self.idx2sym.append(sym);

        if(self.sort_vocab):
            sorted_vocab= sorted(self.sym2count.items(), key= lambda item: item[1])[::-1];
            for i in range(len(sorted_vocab)):
                sym = sorted_vocab[i][0];
                self.sym2idx[sym]= i;
                self.idx2sym.append(sym);

        # Add special symbols to vocab
        # Adding an unknown token for unknown words
        #for sym in (['<start>','<eos>','<unk>']):
        #    if dataset not in ['text8', 'enwik8']:
        #        if(self.sym2count.get(sym,0)==0):
        #            self.sym2count[sym]= self.sym2count.get(sym,0) +1;
        #            self.sym2idx[sym]= len(self.idx2sym);
        #            self.idx2sym.append(sym);

    def get_sym(self, index):
        assert 0 <= index < len(self) , 'Error: querying index out of range';
        return self.idx2sym[index];

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices];

    def _get_file_paths(self, dir_path):
        file_paths=[];
        if(self.dataset in ['wt103']):
            file_paths.append(os.path.join(dir_path, 'train.txt'));
        elif(self.dataset in ['ptb', 'wt2', 'enwik8', 'text8']):
            file_paths.append(os.path.join(dir_path, 'train.txt'));
            file_paths.append(os.path.join(dir_path, 'valid.txt'))
            file_paths.append(os.path.join(dir_path, 'test.txt'))
        else:
            #for lm1b do it later
            pass;

        return file_paths;

    def __len__(self):
        return len(self.idx2sym);


def _add_eos(dataset, tokens):
    if(dataset in ['enwik8','text8']):
        return tokens;
    elif(dataset in ['lm1b']):
        return ['<start>']+tokens +['<eos>'];
    else:
        return tokens + ['<eos>'] ;
        
def _tokenize(dataset, file_path, vocab):
    assert os.path.exists(file_path), 'No such path:{} exists to tokenize'.format(file_path);
    print('Tokenizing ',file_path);

    ids=[];
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line= line.strip();
            if(dataset in ['ptb']):
                line= line.lower();
            tokens= line.split()
            tokens= _add_eos(dataset, tokens)
            for token in tokens:
                sym= token;
                if(sym not in vocab.sym2idx):
                    sym='<unk>';
                ids.append(vocab.sym2idx[sym]);
    
    ids= torch.LongTensor(ids);
    return ids;

class Corpus:
    def __init__(self, dataset, data_path, sort_vocab):
        print('Building corpus');
        self.vocab= Vocab(dataset= dataset, dir_path= data_path, sort_vocab= sort_vocab);

        self.dataset= dataset ;
        self.train_data= _tokenize(dataset= dataset, file_path= os.path.join(data_path, 'train.txt'), vocab= self.vocab);
        self.valid_data= _tokenize(dataset= dataset, file_path= os.path.join(data_path, 'valid.txt'), vocab= self.vocab);
        self.test_data= _tokenize(dataset= dataset, file_path= os.path.join(data_path, 'test.txt'), vocab= self.vocab);
    
    # length of the corpus will be the length of the vocabulary 
    def __len__(self):
        return len(self.vocab);


def _batchify(data_tensor, batch_size):
    n_batches= data_tensor.size(0) // batch_size;
    #remove the last incpmplete batch:
    data_tensor= data_tensor.narrow(0,0,n_batches*batch_size);
    data_tensor= data_tensor.view(batch_size,-1).contiguous()
    return data_tensor;  #batch_size* n_batches

def get_lm_corpus(dataset, data_path, sort_vocab, distributed,rank):
    if(sort_vocab):
        corpus_path= os.path.join(data_path,'data_corpus_sorted.pt');
    else:
        corpus_path= os.path.join(data_path,'data_corpus.pt');
    if(os.path.exists(corpus_path)):
        print('Loading existing data');
        corpus= torch.load(corpus_path);
    else:
        print('Creating a fresh corpus at {}'.format(corpus_path));
        if(distributed):
            if(rank==0):
                corpus= Corpus(dataset, data_path, sort_vocab);
                torch.save(corpus, corpus_path);
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
            else:
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                corpus = torch.load(corpus_path);
        else:
            corpus = Corpus(data_path,sort_vocab);
            torch.save(corpus, corpus_path);
    
    return corpus;

def _get_device_data(data, device_batch_size, rank):
    slice_range= slice(
            device_batch_size*rank,
            device_batch_size*(rank+1)
        );
    return data[slice_range, :];

#returns train valid and test data
def get_all_data(corpus, batch_size, eval_batch_size, distributed,rank, world_size):
    train_data= _batchify(corpus.train_data,batch_size)
    valid_data=  _batchify(corpus.valid_data,eval_batch_size)
    test_data=  _batchify(corpus.test_data,eval_batch_size) 

    if(distributed):
        assert batch_size%world_size == 0, 'batch_size not divisible by world_size'
        device_batch_size= batch_size// world_size;
        device_eval_batch_size= eval_batch_size// world_size;
        train_data= _get_device_data(train_data, device_batch_size, rank);
        if(world_size<=2):
            #since the eval batch size=10 it wont be divisible by world size =4 so doing like this
            assert eval_batch_size%world_size == 0, 'eval_batch_size not divisible by world_size'
            valid_data= _get_device_data(valid_data, device_eval_batch_size, rank);
            test_data= _get_device_data( test_data, device_eval_batch_size, rank);

    return train_data, valid_data, test_data ;

#tgt_len is the block_size and vice versa
def get_this_iter_data(corpus_data, start_pos, tgt_len, device):
    data= corpus_data[:, start_pos: start_pos+tgt_len].contiguous().to(device);
    target= corpus_data[:, start_pos+1: start_pos+tgt_len+1].contiguous().to(device);
    start_pos+= tgt_len;
    #print('data_process: start_pos',start_pos, corpus_data.size(1), tgt_len);
    if(start_pos+1 >= corpus_data.size(1)-tgt_len):
        #whwnever start_pos==0 we complete one epoch
        start_pos=0 ;

    return data, target, start_pos;
    

def get_custom_data(corpus, batch_size, file_name, distributed,rank, world_size,
                                 dataset,):
    custom_data= _tokenize(dataset= dataset, path= file_name, vocab= corpus.vocab);
    custom_data= _batchify(custom_data, batch_size);
    return custom_data;

