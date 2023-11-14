 #visualisation.py

import matplotlib;
matplotlib.use('Agg');
import torch;
import os;
import math;
import torch.distributed as dist;
import seaborn as sns;
import matplotlib.pyplot as plt;

def save_matrix(attn_matrix, count):
    #Only the rank 0 process will save the matrices
    if(dist.get_rank()>0):
      return ;
    dir_path= 'visualisation'
    if(not os.path.exists(dir_path)):
        os.makedirs(dir_path);

    checkpoint_path= os.path.join(dir_path,str(count)+'.pt')
    checkpoint_state={};
    if checkpoint_path:
        checkpoint_state = {
            'attn_matrix': attn_matrix ,
        };
    torch.save(checkpoint_state, checkpoint_path)


def load_matrix(matrix_no, distributed= True):
    if(dist.get_rank()>0):
      return ;
    dir_path= 'visualisation'
    checkpoint_path= os.path.join(dir_path,str(matrix_no)+'.pt')
    if(not os.path.exists(checkpoint_path)):
        return None;

    if distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        checkpoint_state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(checkpoint_path)
    attn_matrix = checkpoint_state['attn_matrix'];
    return attn_matrix;




# returns the last genrated line in the file
def load_last_generated():
    path= os.path.join('visualisation','generated.txt');
    with open(path) as f:
        for line in (f.readlines()[-1:]):
            return line;

def convert_generated(corpus):
    path= os.path.join('visualisation','generated.txt');
    f= open(path, "r");
    #Reading file line by line
    print('GENERATED TEXT:')
    for line in f:
        print(corpus.vocab.get_symbols(line));


def plot_fig(data,x,y):
    # We can use gist_gray or binary as cmap ailso
    sns.heatmap(data,cmap='gist_gray', xticklabels=x, yticklabels=y, vmin=0.0, vmax=1.0, cbar= False);

#filename should be like 1 and we append'.png' to it
def visualise(attn_matrix, x,y, file_name):
    filename= str(file_name)+ '.png';
    dir_path= 'visualisation';
    fig_path= os.path.join(dir_path,filename) ;
    print('attn_matrix shape',attn_matrix.shape);
    plot_fig(data= attn_matrix,x=x,y=y);
    ###################################################
    #plt.imshow(attn_matrix,vmin=0.0, vmax=1.0);
    ###################################################
    plt.savefig(fig_path);


def visualise_data(model,test_data, n_layers,n_heads, eval_tgt_len, eval_mem_len, corpus, 
                   rank, distributed= True):
    if(rank>0):
      return ;

    print('First execute test mode then this mode gives correct results');
    dir_path= 'visualisation';
    model_name='mymodel';

    query_start= math.ceil(eval_mem_len/ eval_tgt_len)* eval_tgt_len;
    K= test_data[0, query_start+ eval_tgt_len- eval_mem_len: query_start+ eval_tgt_len];
    Q= test_data[0, query_start: query_start+eval_tgt_len]
    print('Q.shape',Q.shape,'k.shape',K.shape);
    x = corpus.vocab.get_symbols(K);
    y= corpus.vocab.get_symbols(Q); 
    
    # Since image_id starts with 1 so we add 1 below 
    start_head_no= math.ceil(eval_mem_len/ eval_tgt_len)* n_layers +1 ;  
    last_head_no= (math.ceil(eval_mem_len/ eval_tgt_len)+1)* n_layers +1
    # WE increment by as we want to see onlty the Q+content bias attn_matrices
    for count in range(start_head_no, last_head_no,2):
        attn_matrix= load_matrix(matrix_no = str(count), distributed= distributed);
        print('visualise data attn_matrix.shape',attn_matrix.shape);
        print('count', count);
        for head_no in range(n_heads):
            image_id= count*n_heads -(8-(head_no+1));
            print('image id', image_id);
            visualise(attn_matrix[0][head_no], x,y, str(image_id));

    

# Call this function inside adaptive_io or models
def generate(model,corpus,tgt_len, n_layer, mem_len=512,hidden_size=512,eval_tgt_len= 128,
             start= 'the', max_seq_len=2000, device='cuda', device_eval_batch_size=10):
    
    model.eval()
    mems= [torch.zeros(device_eval_batch_size ,mem_len , hidden_size).to(device)
                for layer in range(n_layer)]
    context, actual_text= [], [];
    start_pos=0; actual_batches=0;
    max_batches_possible= math.ceil(eval_data.size(1)/ eval_tgt_len);
    
    with torch.no_grad():
        for i in range(max_batches_possible):
            actual_batches+=1;
            data, target, start_pos= get_this_iter_data(corpus_data= eval_data, start_pos=start_pos,
                                                        tgt_len= eval_tgt_len, device= device);
            actual_text.append(data.detach());
            _, mems,_ = model(x=data, target= target, mems= mems, mode= mode);
            
            if(i>10):
                pass;


    with torch.no_grad():
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if(i> mem_len+max_seq_len):
                break;
            actual_text.append(data.detach());

        for i, (data, target, seq_len) in enumerate(eval_iter):
            print('data');
            if(i>= mem_len):
                input= data.detach();
                break;
            data, target= data.t().contiguous().to(device), target.t().contiguous().to(device);
            original_loss, mems, mean_cos_loss = model(x=data, target= target,mem= mems, mode= 'test')
            context.append(data[0,-1].detach());
            
            
        for _ in range(max_seq_len):
            print('input shape', input.shape);
            target= torch.zeros(eval_batch_size,input.size(1)).to(device);
            input= input.to(device);
            target= torch.zeros(eval_batch_size,input.size(1)).to(device);
            _, mems, _ = model(x=input, target= target,mem= mems, mode='generate');
            input= load_last_generated();
            input =torch.tensor([input])[None, :];
    
    print("CONTEXT:")
    for i in range(context):
        print(corpus.vocab.get_symbols(i));
    print("ACTUAL TEXT:")
    for i in range(actual_text):
        print(corpus.vocab.get_symbols(i));
    print("GENERATED TEXT:")
    convert_generated(corpus)
    model.train();


# takes the index of the generated symbol and saves it 
def save_generated(ind):
    path= os.path.join('visualisation','generated.txt');
    with open(path, 'a+') as f:
            f.write(s + '\n');

