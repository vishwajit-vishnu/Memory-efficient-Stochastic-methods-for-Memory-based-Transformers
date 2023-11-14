# version2 of trainer.py

# trainer.py 
import torch;
import os, time;
import math;
import itertools;
from utils import save_checkpoint, load_checkpoint ;
from visualisation import save_matrix ;
from data_process import get_this_iter_data;

def format_log(loss, split, dataset):
    if(loss>50):
        return '|{} loss too big: {} |'.format(split,loss);
    if dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
            split, loss, loss / math.log(2))
    else:
        log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
            split, loss, math.exp(loss))
    return log_str


# HERE eval_iter is just the Valid_data/ test_data 
def evaluate(model,eval_data, eval_tgt_len, mem_len, device_eval_batch_size,hidden_size, n_layer,device, 
             distributed, world_size, logging , rank, mode= None, prune_elements= None):
  
    model.eval();
    print('eval _data shape', eval_data.shape);
    mems= [torch.zeros(eval_data.size(0) ,mem_len , hidden_size).to(device)
                for layer in range(n_layer)]

    total_loss = 0.0 
    start_pos=0; actual_batches=0;
    max_batches_possible= math.ceil(eval_data.size(1)/ eval_tgt_len);
    
    with torch.no_grad():
        while(True):
            actual_batches+=1;
            data, target, start_pos= get_this_iter_data(corpus_data= eval_data, start_pos=start_pos,
                                                        tgt_len= eval_tgt_len, device= device);

            #print('Evaluate:data.device', data.device, target.device);
            #print('Evaluate:',mode,' data:{}, target={}, start_pos {}'.format(data.shape, target.shape, start_pos));
            original_loss, mems, _ = model(x=data, target= target, mem= mems, mode= mode, 
                                            prune_elements= prune_elements) ;
            total_loss+= float(original_loss);
            if(start_pos==0):
                break;

    total_loss= total_loss/float(actual_batches);
    if(distributed):
        temp= torch.tensor([total_loss]).to(device);
        print('Eval:',mode,' loss before sync', temp[0],temp.device, 'world_sz',world_size,'actual_batches', actual_batches);
        #torch.distributed.all_reduce(temp);
        torch.distributed.reduce(temp,0);
        #print("eval reduced_total loss", temp, 'world_size', world_size);
        total_loss= temp[0]/world_size;

    model.train();
    print('evaluation done');
    return float(total_loss)

#####   trainv2()

def train(model,train_data,valid_data, hidden_size, n_layer, device,optimizer,mem_len, device_batch_size, 
          batch_chunk, device_eval_batch_size,scheduler,scheduler_name, grad_clip, log_interval, 
          warmup_step, eval_interval, work_dir, max_step, eval_tgt_len, logging, dataset, distributed, 
          world_size, rank, tgt_len, learning_rate, train_mode= None, corpus=None):
    
    # Turn on training mode which enables dropout.
    model.train();
    print('train_data_shape', train_data.shape,'valid_data.shape', valid_data.shape);

    val_loss= 1000000.0; 
    # Load the best model saved and start training from that point
    iter_init, best_val_loss = load_checkpoint(checkpoint_path= work_dir, model= model, optimizer= optimizer, 
                           scheduler= scheduler, distributed= distributed);
    train_step= iter_init;
    
    mems= [torch.zeros(device_batch_size,mem_len, hidden_size).to(device)
            for layer in range(n_layer)]

    for epoch in itertools.count(start=1):
        train_loss=0.0; log_start_time= time.time();
        start_pos= 0; n_steps_current_interval= 0;
        
        while(train_step< max_step):
            n_steps_current_interval+=1;
            data, target, start_pos= get_this_iter_data(corpus_data=train_data, start_pos= start_pos,
                                                        tgt_len= tgt_len, device= device);
            #print('Train: data.shape', data.shape,'target.shape', target.shape,'start_ps', start_pos);
            model.zero_grad();
            loss= 0.0; original_loss=0.0 ;
            temp_mem=[];
            
            #chunking has not been tested yet
            if batch_chunk > 1:
                assert device_batch_size % batch_chunk ==0 , 'chunk size does not divide batch_size'
                split_size= device_batch_size//batch_chunk;
                data_chunks = torch.chunk(data, batch_chunk)
                target_chunks = torch.chunk(target, batch_chunk);
                for i in range(batch_chunk):
                    split_val= slice(i*split_size, (i+1)*split_size)
                    data_i = data_chunks[i].contiguous()
                    target_i = target_chunks[i].contiguous();
                    mem_chunk_i= [m[split_val,:,:] for m in mems];
                    original_current_loss, mem_chunk_i, aux_current_loss= model(x= data_i, target= target_i, mem= mem_chunk_i,
                                                                                mode='train',  train_mode= train_mode);
                    original_loss = original_current_loss / batch_chunk ;
                    aux_loss= mean_aux_loss/ batch_chunk;
                    current_loss= original_current_loss + mean_current_cos_loss;
                    loss= current_loss;
                    # loss += current_loss
                    model.zero_grad();
                    loss.backward(retain_graph= True);
                    temp_mem.append(mem_chunk_i);
                    train_loss += float(original_loss);

                mems = [torch.cat([temp_mem[i][l] for i in range(batch_chunk)], 
                                  dim=0) for l in range(len(mems))];
                # loss.backward(retain_graph=True);
                loss=0.0;

            else:
                if(corpus is not None):
                    print('data',corpus.vocab.get_symbols(data[0]));
                original_loss, mems, aux_loss= model(x= data, target= target,mem= mems, mode= 'train', train_mode= train_mode);
                loss = original_loss + aux_loss ;

                loss.backward();
                train_loss += float(original_loss)

            if(grad_clip>0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # step-wise learning rate annealing
            train_step += 1

            if (scheduler_name.lower() =='cosine' and scheduler is not None):
                # linear warmup stage
                if train_step <= warmup_step:
                    curr_lr = learning_rate * train_step / warmup_step;
                    optimizer.param_groups[0]['lr'] = curr_lr;
                    scheduler.step();
                    #scheduler.step(train_step);
                else:
                    scheduler.step() ;

            elif scheduler_name.lower() == 'lambdalr' and scheduler is not None:
                scheduler.step(train_step);

            else:
                pass;
            
            if train_step % log_interval == 0:
                cur_loss = train_loss / n_steps_current_interval
                elapsed = time.time() - log_start_time
                log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} |'.format(
                    epoch, train_step, n_steps_current_interval, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / log_interval);
                
                if(distributed):
                    temp= torch.tensor([cur_loss]).to(device);
                    torch.distributed.all_reduce(temp);
                    #torch.distributed.reduce(temp,0);
                    total_loss= temp[0]/world_size;
                if(rank==0):
                    log_str += format_log(cur_loss,'train', dataset);
                    logging(log_str);

                n_steps_current_interval=0;
                train_loss = 0.0
                log_start_time = time.time()

            if train_step % eval_interval == 0:
                eval_start_time= time.time();				
                val_loss = evaluate(model= model,eval_data= valid_data, eval_tgt_len= eval_tgt_len, mem_len= mem_len, 
                                    device_eval_batch_size= device_eval_batch_size, hidden_size= hidden_size, 
                                    n_layer= n_layer, device= device,distributed= distributed , world_size= world_size, 
                                    logging= logging, rank= rank, mode= 'val' );
                
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s | '.format(train_step // eval_interval, 
                                                                            train_step, (time.time() - eval_start_time));
                log_str+= format_log(val_loss,'val', dataset);

                if((rank==0 and distributed) or not distributed):
                    logging(log_str);
                    logging('-'*100);
                
                # Save the model if the validation loss is the best we've seen so far.
                if ( val_loss < best_val_loss):
                    save_checkpoint(checkpoint_path= work_dir, iter_no = train_step, model= model,
                        optimizer= optimizer, scheduler= scheduler,best_val_loss= val_loss,distributed= distributed, rank= rank);
                    best_val_loss = val_loss

                if scheduler_name.lower() == 'plateau'and scheduler is not None:
                    scheduler.step(val_loss)

            if(start_pos==0):
                break;
        if(train_step>=max_step):
            break;

    if((rank==0 and distributed) or not distributed):
        logging('-' * 100)
        logging('End of training after reaching max steps')


#### manual check done #############
