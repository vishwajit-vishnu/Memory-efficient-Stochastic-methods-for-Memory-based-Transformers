#pruning.py

import torch;
from trainer import evaluate;
from trainer import format_log;

def prune(model, n_layer, n_heads,eval_data, eval_tgt_len, mem_len, device_eval_batch_size,
          hidden_size, device, distributed, world_size, logging , rank, dataset ):
  
    logging("STARTING PRUNING EXPERIMENTS");
    layer_no , head_no = 0,0 ;

    while(layer_no < n_layer):
        head_no =0;
        while(head_no < n_heads):
            if(rank==0):
                logging('pruning layer:{} head no: {}'.format(layer_no+1, head_no+1));
            prune_scale= torch.ones(n_heads);
            prune_scale[head_no]=0;
            prune_elements=[layer_no, prune_scale];
            print(prune_elements);
            # pass to the model and evaluate
            # import the evaluate function
            test_loss= evaluate(model= model,eval_data= eval_data, eval_tgt_len= eval_tgt_len, mem_len= mem_len, 
                     device_eval_batch_size= device_eval_batch_size,hidden_size= hidden_size,
                      n_layer= n_layer,device= device, distributed= distributed, world_size= world_size,
                      logging= logging , rank= rank, mode='test' , prune_elements= prune_elements);
            log_str= format_log(test_loss, 'test', dataset);
            if(rank==0):
                logging('#'*100);
                logging(log_str);   logging('*'*100);

            head_no +=1 ;
        layer_no +=1;
            
