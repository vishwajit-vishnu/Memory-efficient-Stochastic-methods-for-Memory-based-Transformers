#### utils.py

import sys;
import math;
import os;
import torch;

### Setup env  ###########


def setup_env(distributed_init_method, local_rank):
    assert torch.cuda.is_available(), ' cuda not available' ;
    torch.distributed.init_process_group(
        backend='nccl',
        init_method= distributed_init_method,
    );
    rank= torch.distributed.get_rank();
    world_size= torch.distributed.get_world_size();
    print("rank: ",rank," local_rank:",local_rank," world_size",world_size);
    torch.cuda.set_device(local_rank);
    return rank,world_size;
    


########################################
##### Saving and loading checkpoints
########################################
def _load_checkpoint(checkpoint_path, model, optimizer, scheduler,distributed):
    #print('loading from a checkpoint at {}'.format(checkpoint_path));
    checkpoint_path= os.path.join(checkpoint_path, 'saved_model.pt');
    if(not os.path.exists(checkpoint_path)):
        return 0, 10000.0;
    if distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        #checkpoint_state = torch.load(checkpoint_path);
        # New Line added above
        checkpoint_state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(checkpoint_path)
    iter_init = checkpoint_state['iter_no'] + 1  # next iteration
    best_val_loss= checkpoint_state['best_val_loss'];
    model.load_state_dict(checkpoint_state['model'])
    optimizer.load_state_dict(checkpoint_state['optimizer'])
    if ('scheduler_iter' in checkpoint_state and scheduler is not None):
        #scheduler.load_state_dict(checkpoint_state['scheduler']);
        # we only need the step count
        scheduler.step(checkpoint_state['scheduler_iter'])
    return iter_init, best_val_loss;



def load_checkpoint_trainparam(checkpoint_path, distributed):
    checkpoint_path= os.path.join(checkpoint_path, 'trainparam.pt');
    if(not os.path.exists(checkpoint_path)):
        return 0;
    if distributed:
        # the model is saved from gpu0 so we need to map it to CPU first
        checkpoint_state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        checkpoint_state = torch.load(checkpoint_path)
    train_step = checkpoint_state['train_step'];
    return train_step;

def save_checkpoint_trainparam(checkpoint_path, train_step, distributed, rank):
    checkpoint_path= os.path.join(checkpoint_path, 'trainparam.pt');
    checkpoint_state={};
    if checkpoint_path:
        checkpoint_state = {
            'train_step': train_step,
        };
    if((distributed and rank==0) or not distributed):
        torch.save(checkpoint_state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, distributed):
    if (checkpoint_path and os.path.exists(checkpoint_path)):
        return _load_checkpoint(checkpoint_path=checkpoint_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                distributed=distributed)
    return 0, 100000.0;


def save_checkpoint(checkpoint_path, iter_no, model,
                    optimizer, scheduler, best_val_loss, distributed, rank):
    checkpoint_path= os.path.join(checkpoint_path, 'saved_model.pt');
    checkpoint_state={};
    if checkpoint_path:
        checkpoint_state = {
            'iter_no': iter_no,  # last completed best model's iteration
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if scheduler is not None:
            checkpoint_state['scheduler_iter'] = scheduler.last_epoch;
            #checkpoint_state['scheduler'] = scheduler;
        if((distributed and rank==0) or not distributed):
            torch.save(checkpoint_state, checkpoint_path)

