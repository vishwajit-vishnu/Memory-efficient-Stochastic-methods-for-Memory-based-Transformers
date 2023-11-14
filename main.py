
##############  main.py    ###########

import math;
import argparse;
import os,sys, time;
import torch;
import numpy as np;
import itertools;

from data_process import get_lm_corpus, get_all_data, get_this_iter_data ;
from exp_utils import create_exp_dir;
from utils import  load_checkpoint, setup_env, load_checkpoint_trainparam;
from optim import get_head_sizes, get_head_sizes_v2;
from optim import get_optimizer_and_scheduler;
from trainer import train, evaluate, format_log;
from models import Transformer;
from weight_initialisation import weights_init;

from visualisation import visualise_data;
from pruning import prune;

parser= argparse.ArgumentParser(description= 'Transformer by- vishnu');

################################# DATA  ############################
parser.add_argument('--data', type= str, help='dataset directory');
parser.add_argument('--dataset', type=str, default='enwik8',
                    choices=[ 'enwik8', 'text8', 'wt103','wt2','lm1b'], 
                    help='dataset name');
parser.add_argument('--tgt_len', type=int, default=512,
                    help='n_tokens to predict');
parser.add_argument('--eval_tgt_len', type=int, default= 128,
                    help='n_tokens to predict during eval time');
parser.add_argument('--batch_size', type=int, default=60, help='batch_size')
parser.add_argument('--batch_chunk', type=int, default=1,
                   help='split batches into chunks to save memory');
parser.add_argument('--eval_batch_size', type=int, default=10, 
                    help='eval batch_size')

########################  MODEL Params  ############################
parser.add_argument('--n_layer', type=int, default=12,
                    help='total no of layers');
parser.add_argument('--hidden_size',type=int, default=512,
                    help= 'hidden size');
parser.add_argument('--embedding_size', type=int, default=128,
                    help= 'embedding dimension');
parser.add_argument('--n_heads', type=int, default=8,
                    help= 'no of attention heads');
parser.add_argument('--inner_hidden_size', type=int, default= 2048,
                    help= 'size of feedforward network');
parser.add_argument('--dropout', type=float, default=0.0,
                    help= 'global dropout rate');
parser.add_argument('--emb_dropout', type=float, default=0.0,
                    help= 'dropout for embedding');
parser.add_argument('--dropatt', type=float, default=0.0,
                    help= 'attn dropout rate');
parser.add_argument('--mem_len', type= int, default=256,
                    help='memory length');
parser.add_argument('--ext_len', type= int, default=0,
                    help='extended length'); # Not using this 
parser.add_argument('--activation_fn', type=str, default='relu', choices=['relu', 'gelu', 'selu'],
                    help= 'activation function in feedforward layer');
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking');
parser.add_argument('--learn_bias_scales', action='store_true', default= False,
                    help='learn bias scales for positional and content biases');
parser.add_argument('--factorized_embedding', action='store_true', default= False,
                    help= 'Use facctorized/decomposed embedding matrices')
parser.add_argument('--NL_ff_input', action='store_true', default= False,
                    help= 'Input non linear representation to FF layer after doing layernorm');
parser.add_argument('--learn_head_importance', action='store_true', default= False,
                    help='Learn head importances');


# cross_head_attention       ***************
parser.add_argument('--cross_head_attn', action= 'store_true', default= False,
                    help= 'Use cross_head attention');
parser.add_argument('--ch_prob', type= float, default= 0.0,
                    help= 'attend cross head attn with this probability');

# transformations        ******************
parser.add_argument('--rotation', action='store_true', default= False,
                    help='randomly rotate the input vectors');
# parser.add_argument('--add_noise', action='store_true', default= False,
#                     help='randomly add some Normal(0,1) noise to the input vectors');
# parser.add_argument('--noise_std', type= float, default=1.0,
#                     help= 'noise N(0,1) is scaled by this value');

# arguments for similarity_loss *************
parser.add_argument('--similarity_loss', action= 'store_true', default= False,
                    help='activate cosine similarity loss');
parser.add_argument('--alpha_similarity',  type=float, default=0.9,
                    help='alpha for scaling cosine similarity loss');

# adaptive io params below   ****************
parser.add_argument('--adapt_io', action='store_true', default= False,
                    help='Use adaptive io');
parser.add_argument('--adapt_io_divval', type=int, default=4,
                    help='dimension division value');
parser.add_argument('--adapt_io_cutoffs',default= [20000,20000,200000],
                      help='cutoff values');  
parser.add_argument('--adapt_io_tied',action='store_true',default= False ,
                      help='tie the input parameters with the output parameters');

# variable size heads   *****************
parser.add_argument('--variable_size_heads', action='store_true' , default=False,
                    help='Use the variable size heads');
parser.add_argument('--vh_version', default='v1', choices=['v1','v2'],
                    help='which version of variable size attention heads to use');

# skip retain training mechanism parameters***************
parser.add_argument('--skip_retain', action='store_true', default= False,
                    help='Use skip ret  ain training mechanism with functional_probability')
parser.add_argument('--uniform_skip_prob', type= float, default=0.0, 
                    help= 'Use skip retain mechanism with uniformly skipping every layer');
parser.add_argument('--learn_skip_prob', action='store_true', default=False,
                    help='Use skip retain training by learning each layers skip probability');

# prenorm and share weights ****************
parser.add_argument('--prenorm', action='store_true', default= False,
                    help=' use prenormalization');
parser.add_argument('--share_weight', action= 'store_true', default= False,
                    help='Whether to share weight in the transformer layers');
parser.add_argument('--layer_chunk_size', type= int, default=2,
                    help='This tells how many consecutive transformer layers share weights at once');

####################  Optimizer/ Scheduler Params  ############################
parser.add_argument('--optim',type=str, default='adam',
                    choices=['adam','adagrad','amsgrad','sparseadam','adamw'],
                    help='optimiser to use');
parser.add_argument('--lr', type=float, default=0.007,
                    help='learning rate');
parser.add_argument('--scheduler', type=str, default='lambdalr',
                    choices=['cosine', 'lambdalr','plateau'],
                    help='scheduler to use');
parser.add_argument('--warmup',type=int, default=4000,
                    help='no of warmup steps');
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler');
parser.add_argument('--decay_rate', type=float, default=0.2,
                    help='decay factor when plateau');
parser.add_argument('--grad_clip', type=float, default=0.25,
                    help='gradient clipped to this value');
parser.add_argument('--patience', type=int, default=3,
                    help='patience for reduceLronplateau scheduler');

######################## Train params ########################
parser.add_argument('--log_interval', type=int, default=200,
                    help='steps after which the logging is done');
parser.add_argument('--eval_interval', type=int, default=600,
                    help='Steps after which the evaluation is done');
parser.add_argument('--mode', type=str, choices=['train', 'test','visualise','generate', 'reflect','prune'],
                    help='Is it training or testing phase');
parser.add_argument('--train_mode', type=str, choices=['freeze_pos','finetune'], default= None,
                    help='Freeze the relative positions');    
parser.add_argument('--apply_weights', action='store_true', default= False,
                    help='apply the weight initialisation scheme')
parser.add_argument('--random_relpos', action='store_true', default= False,
                    help='add random shift to relative positional encodings');

####################  Other params  ############################
parser.add_argument('--device', type=str, default='cuda',
                    choices=['cuda','cpu'], help='device to use')
parser.add_argument('--seed', type=int ,default=1111,
                    help='random seed');
parser.add_argument('--distributed',action='store_true',
                    help='use distributed multigpu');
parser.add_argument('--work_dir', default='LM',
                    help='store all details in this work dir');
parser.add_argument('--max_steps',type=int,default=550000,
                    help='max no of epochs');
parser.add_argument('--distributed_init_method', type=str,
                    default='env://',help='something like tcp://127.0.0.1:23456');
parser.add_argument('--local_rank', type=int, default=0, help=' local rank of gpu');
parser.add_argument('--cuda_no', type= int, default=0, help='which cuda device to ude');

args= parser.parse_args();

args.work_dir= '{}-{}'.format(args.work_dir,args.dataset);
logging= create_exp_dir(args.work_dir,scripts_to_save=['main.py', 'models.py'])


np.random.seed(args.seed); torch.manual_seed(args.seed);
device= 'cuda' if args.device=='cuda' else 'cpu';
if(not args.distributed and torch.cuda.is_available()):
    device= device +':' +str(args.cuda_no) ;


if(torch.cuda.is_available()):
    torch.cuda.empty_cache();
    args.device_name= torch.cuda.get_device_name(torch.device(device));
    torch.cuda.manual_seed_all(args.seed);

device= torch.device(device);
##### setup distributed environment
#####################################
if(args.distributed):
    args.rank, args.world_size= setup_env(args.distributed_init_method, args.local_rank);
else:
    args.rank, args.world_size = args.cuda_no, 1;

############### SEtting max threads to use omit to use all resources

torch.set_num_threads(10);   
#########################################
### Data Loading
#########################################
args.sort_vocab= True;
if(args.dataset=='lm1b'):
    args.sort_vocab= False

corpus= get_lm_corpus(dataset= args.dataset, data_path= args.data, sort_vocab= args.sort_vocab, 
                      distributed= args.distributed, rank= args.rank);
args.n_tokens = len(corpus.vocab)

if(args.distributed):
    args.device_batch_size= args.batch_size // args.world_size
    args.device_eval_batch_size= args.eval_batch_size // args.world_size ;
    if(args.rank==0):
        logging('Using distributed with batchsize per world:{}, world_size:{}'.format(args.device_batch_size, args.world_size));
else:
    args.device_batch_size= args.batch_size;
    args.device_eval_batch_size= args.eval_batch_size;

#########################################################
train_data, valid_data, test_data= get_all_data(corpus= corpus, batch_size= args.batch_size, 
                                        eval_batch_size= args.eval_batch_size,
                                        distributed= args.distributed, rank= args.rank, world_size= args.world_size);

#print('all_train_datashape', train_data.shape, valid_data.shape, test_data.shape);
############# Adaptive embedding softmax cutoffs ############
if args.dataset == 'wt103':
    args.adapt_io_cutoffs = [20000, 40000, 200000]
elif args.dataset == 'lm1b':
    args.adapt_io_cutoffs = [60000, 100000, 640000]
# For character level LM no need of using adapt_io
elif (args.dataset =='enwik8'):
    args.adapt_io_cutoffs = [202] ;
elif (args.dataset =='text8'):
    args.adapt_io_cutoffs= [26] ;
##############################################################
#### Build the model 
##############################################################


if(args.variable_size_heads):
    if(args.vh_version=='v1'):
        head_sizes_list= get_head_sizes(args.n_heads,args.hidden_size, args.variable_size_heads);
    else:
        head_sizes_list= get_head_sizes_v2(args.n_heads,args.hidden_size, args.variable_size_heads);
else:
    head_sizes_list= None;   #Use equal size heads
                  

model= Transformer(vocab_size= args.n_tokens, hidden_size= args.hidden_size, embedding_size= args.embedding_size,
                   n_heads= args.n_heads, n_layer= args.n_layer, mem_len= args.mem_len, 
                   inner_hidden_size= args.inner_hidden_size, emb_dropout= args.emb_dropout, dropout= args.dropout,
                   activation_fn= args.activation_fn, dropatt= args.dropatt, adapt_io_cutoffs= args.adapt_io_cutoffs,
                   adapt_io_divval= args.adapt_io_divval, adapt_io_tied= args.adapt_io_tied, 
                   variable_size_heads= args.variable_size_heads, head_sizes_list= head_sizes_list,
                   clamp_len= args.clamp_len, same_length= args.same_length, adapt_io=args.adapt_io, 
                   skip_retain= args.skip_retain, uniform_skip_prob= args.uniform_skip_prob,
                   factorized_embedding= args.factorized_embedding, NL_ff_input= args.NL_ff_input,
                   learn_bias_scales= args.learn_bias_scales, prenorm= args.prenorm, 
                   share_weight= args.share_weight, layer_chunk_size= args.layer_chunk_size,
                   cross_head_attn= args.cross_head_attn, ch_prob= args.ch_prob, 
                   learn_head_importance= args.learn_head_importance, rotation= args.rotation,
                   random_relpos= args.random_relpos).to(device);



if(args.apply_weights):
    model.apply(weights_init);

if((args.rank==0 and args.distributed) or not args.distributed):
    logging('head_sizes:'+str(head_sizes_list));
    logging('Initialised the model');

if (args.distributed):
    assert torch.distributed.is_available(), 'Distributed not available' ;
    local_rank= args.local_rank; print('distributed local_rank',local_rank);
    model= model.to(device);
    model= torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device= local_rank,
                                                     find_unused_parameters=True);

optimizer, scheduler= get_optimizer_and_scheduler(model= model, learning_rate= args.lr, 
                                                  optim= args.optim,lr_warmup= args.warmup, 
                                                  scheduler= args.scheduler, eta_min= args.eta_min,
                                                  patience= args.patience, decay_rate= args.decay_rate);

#Before training load if the model is saved
iter_init, best_val_loss = load_checkpoint(checkpoint_path= args.work_dir, model= model, optimizer= optimizer, 
                           scheduler= scheduler, distributed= args.distributed);

if(iter_init==0):
    trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad);
    total_params =sum(p.numel() for p in model.parameters());
    # args.nonemb_params= sum([p.nelement() for p in model.layers.parameters()]);
    args.n_all_param = total_params; 
    if((args.distributed and args.rank==0) or not args.distributed):
        logging('='*100);
        for k,v in args.__dict__.items():
            logging(' - {}: {}'.format(k,v));
        logging('total_params: {} million'.format(total_params/1e6));
        logging('trainable paramd: {}'.format(trainable_params/1e6));
        logging('='*100);

#################################################
######## Different modes
################################################

# Reflect mode just shows how we processes the data and prints it
if(args.mode== 'reflect'):
    try:
        temp_start_pos= 0; logging('train_data');
        for _ in range(20):
            data, target, temp_start_pos= get_this_iter_data(corpus_data= train_data, 
                                                             start_pos= temp_start_pos, tgt_len= args.tgt_len, device= args.device);
            print('data.shapein train', data.shape, target.shape, temp_start_pos);
            if(args.dataset in ['enwik8', 'text8']):
                mystr= chr(corpus.vocab.get_symbols(data[0]))
            else:
                mystr= corpus.vocab.get_symbols(data[0])

            mystr='train_ data'+ str(mystr);
            if(args.rank==0):
                logging(mystr);
        
        temp_start_pos= 0; logging('valid data')
        for _ in range(20):
            data, target, temp_start_pos= get_this_iter_data(corpus_data= valid_data, 
                                          start_pos= temp_start_pos, tgt_len= args.tgt_len, device= args.device);
            print('data shape', data.shape, target.shape, temp_start_pos);
            if(args.dataset in ['enwik8', 'text8']):
                mysstr= chr(corpus.vocab.get_symbols(data[0]))
            else:
                mystr= corpus.vocab.get_symbols(data[0])
            if(args.rank==0):
                mystr= 'valid_data'+ str(mystr);
                logging(mystr);

        temp_start_pos= 0; logging('test data')
        for _ in range(20):
            data, target, temp_start_pos= get_this_iter_data(corpus_data= test_data,
                                              start_pos= temp_start_pos, tgt_len= args.tgt_len, device= args.device);
            if(args.dataset in ['enwik8', 'text8']):
                mystr= chr(corpus.vocab.get_symbols(data[0]))
            else:
                mystr= corpus.vocab.get_symbols(data[0]);
            if(args.rank==0):
                mystr= 'test_data'+ str(mystr);
                logging(mystr);

    except Exception as e:
        # raise e; 
        logging('EXCEPTION in reflecting data: {}'.format(e));
    except:
        logging('reflecting data ERROR: {}'.format(sys.exc_info()[0]));
        logging('='*100);


if(args.mode=='train' ):
    try:
        train( model= model, train_data= train_data, valid_data= valid_data,
                  hidden_size= args.hidden_size, n_layer= args.n_layer, device= args.device, optimizer= optimizer, 
                  mem_len= args.mem_len, device_batch_size= args.device_batch_size, batch_chunk= args.batch_chunk, 
                  device_eval_batch_size= args.device_eval_batch_size, scheduler= scheduler, 
                  scheduler_name= args.scheduler,
                  grad_clip= args.grad_clip, log_interval= args.log_interval, warmup_step= args.warmup, 
                  eval_interval= args.eval_interval, work_dir= args.work_dir, max_step= args.max_steps, 
                  eval_tgt_len= args.eval_tgt_len, logging=logging, 
                  dataset= args.dataset, distributed= args.distributed,world_size= args.world_size, rank= args.rank, 
                  tgt_len= args.tgt_len, learning_rate= args.lr , train_mode= args.train_mode ,corpus= None);

    except KeyboardInterrupt:
        logging('EXITING from training earlydue to keyboard interrupt')

    except Exception as e:
        print(e); raise e;
        logging('EXCEPTION in training: {}'.format(e));
    except:
        logging('training ERROR: {}'.format(sys.exc_info()[0]));
        logging('='*100);

    # Load the best saved model
    try:
        _, _= load_checkpoint(checkpoint_path= args.work_dir, model= model, optimizer= optimizer, 
                              scheduler= scheduler, distributed= args.distributed);
    except Exception as e:
        logging('EXCEPTION in loading data after training:{}'.format(e));
    except:
        logging('loading data ERROR:{}'.format(sys.exc_info()[0]));
        logging('='*100);

    # Run on test data.
    try:
        test_loss = evaluate(model= model, eval_data= test_data, eval_tgt_len= args.eval_tgt_len, mem_len= args.mem_len, 
                             device_eval_batch_size= args.device_eval_batch_size, hidden_size= args.hidden_size, 
                            n_layer= args.n_layer, device= args.device, distributed= args.distributed, 
                             world_size= args.world_size, logging= logging ,rank= args.rank, mode= None);


        log_str =format_log(test_loss, 'End of training, Test', args.dataset)
        
        if((args.distributed and args.rank==0) or not args.distributed):
            logging(log_str);

    except Exception as e:
        logging('EXCEPTION in testing after training completion {}'.format(e));
    except:
        logging('Testing ERROR: {}'.format(sys.exc_info()[0]));
        logging('='*100);


if(args.mode=='prune'):
    try:
        iter_init, _ = load_checkpoint(checkpoint_path= args.work_dir, model= model, optimizer= optimizer, 
                           scheduler= scheduler, distributed= args.distributed);

    except Exception as e:
        logging('EXCEPTION in testing loading checkpoint: {}'.format(e));
    except:
        logging('loading data ERROR: {}'.format(sys.exc_info()[0]));

    model = model.to(device);
    if((args.rank==0 and args.distributed) or not args.distributed):
        logging("="*100);
        logging('Testing with bsz {} tgt_len {} mem_len {} clamp_len {} same_length:{}'.format(
              args.eval_batch_size, args.tgt_len, args.mem_len, args.clamp_len, args.same_length))
    try:
        prune(model= model, n_layer= args.n_layer,n_heads= args.n_heads, eval_data= test_data, 
              eval_tgt_len= args.eval_tgt_len, 
              mem_len= args.mem_len, device_eval_batch_size= args.device_eval_batch_size, 
              hidden_size= args.hidden_size, device= args.device, distributed= args.distributed, 
              world_size= args.world_size, logging= logging ,rank= args.rank, dataset= args.dataset);


    except Exception as e:
        pddrint(e); raise e;
    except:
        logging('Testing ERROR {}'.format(sys.exc_info()[0]));



if(args.mode=='test'):
    #print("Testing mode selected");
    try:
        iter_init, _ = load_checkpoint(checkpoint_path= args.work_dir, model= model, optimizer= optimizer, 
                           scheduler= scheduler, distributed= args.distributed);
    except Exception as e:
        logging('EXCEPTION in testing loading checkpoint: {}'.format(e));
    except:
        logging('loading data ERROR: {}'.format(sys.exc_info()[0]));
        logging('='*100);

    model = model.to(device)

    if((args.rank==0 and args.distributed) or not args.distributed):
        logging("="*100);
        logging('Testing with bsz {} tgt_len {} mem_len {} clamp_len {} same_length:{}'.format(
              args.eval_batch_size, args.tgt_len, args.mem_len, args.clamp_len, args.same_length))

    try:
        # Run on test data.
        test_loss = evaluate(model= model, eval_data= test_data, eval_tgt_len= args.eval_tgt_len, mem_len= args.mem_len, 
                             device_eval_batch_size= args.device_eval_batch_size, hidden_size= args.hidden_size, 
                            n_layer= args.n_layer, device= args.device, distributed= args.distributed, 
                             world_size= args.world_size, logging= logging ,rank= args.rank, mode='test');
        valid_loss = evaluate(model= model, eval_data= valid_data, eval_tgt_len= args.eval_tgt_len, mem_len= args.mem_len, 
                             device_eval_batch_size= args.device_eval_batch_size, hidden_size= args.hidden_size, 
                            n_layer= args.n_layer, device= args.device, distributed= args.distributed, 
                             world_size= args.world_size, logging= logging ,rank= args.rank, mode='val');

        
        log_str = format_log(valid_loss, 'valid', args.dataset)
        log_str += format_log(test_loss, 'test', args.dataset);
        if((args.rank==0 and args.distributed) or not args.distributed):
            logging('#'*100);
            logging(log_str);

    except KeyboardInterrupt:
        logging('-' * 100)
        logging('Exiting from Testing phase early due to keyboard interrupt');
    
    except Exception as e:
        raise e;
        logging('EXCEPTION in testing :{}'.format(str(e)));

    except:
        logging('Testing ERROR {}'.format(sys.exc_info()[0]));
        logging('='*100);


if(args.mode=='visualise'):
    #print("Visualise mode selected");
    try:
        visualise_data(model= model, test_data= test_data, n_layers= args.n_layer, n_heads= args.n_heads, 
                       eval_tgt_len= args.eval_tgt_len, eval_mem_len= args.mem_len, corpus= corpus, 
                        rank= args.rank, distributed= args.distributed);
    except Exception as e:
        raise e; 
        logging('EXCEPTION in visualising data: {}'.format(e));
    except:
        logging('Visualising data ERROR: {}'.format(sys.exc_info()[0]));
        logging('='*100);


if(args.mode=='generate'):
    try:
        # liad the best saved model
        _, _ = load_checkpoint(checkpoint_path= args.work_dir, model= model, optimizer= optimizer, 
                           scheduler= scheduler, distributed= args.distributed);
        generate(model= model,corpus= corpus,tgt_len= args.tgt_len, n_layer= args.n_layer, mem_len= args.mem_len, 
                 hidden_size=  args.hidden_size, start= 'the', max_seq_len=2000, device=args.device)

    except Exception as e:
        logging('EXCEPTION in Generating data: {}'.format(e));
    except:
        logging('Generating data ERROR: {}'.format(sys.exc_info()[0]));
        logging('='*100);

