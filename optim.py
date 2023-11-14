
### optim.py
### return the optimizer and the scheduler
import torch;


def _get_grad_requiring_params(model):
    grad_requiring_params=[];
    for param in model.parameters():
        grad_requiring_params.append(param);
    return grad_requiring_params;


def _get_optimizer(model, learning_rate, optim):
    if (optim.lower()=='adam'):
        optimizer= torch.optim.Adam(model.parameters(),lr= learning_rate);
    elif(optim.lower()=='adagrad'):
        optimizer= torch.optim.Adagrad(model.parameters(),lr=learning_rate);
    elif(optim.lower()=='amsgrad'):
        optimizer= torch.optim.Adam(model.parameters(),lr= learning_rate,amsgrad= True); #,eps=1e-6
    elif(optim.lower()=='sparseadam'):
        optimizer= torch.optim.SparseAdam(model.parameters(), lr=learning_rate); #eps is default
    elif(optim.lower()=='adamw'):
        optimizer= torch.optim.AdamW(model.parameters(), lr= learning_rate) # eps and weight decay are same
    else:
        optimizer= None;
    return optimizer;

def _get_scheduler(optim,lr_warmup, scheduler , eta_min, patience, decay_rate=0.1):
    if(lr_warmup>0):
        if(scheduler.lower()=='lambdalr'):
            return torch.optim.lr_scheduler.LambdaLR(
                optim,lambda epoch:min(1, epoch/lr_warmup)
            );
        
        elif(scheduler.lower()=='cosine'):
            return torch.optim.lr_scheduler.CosineAnnealingLR(optim, lr_warmup , eta_min)
        elif(scheduler.lower()=='plateau'):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min', patience=patience,
                                                              factor= decay_rate);
        else:
            print('Not written code for the given scheduler. See help');
            return None;
    else:
        return None;


def get_optimizer_and_scheduler(model, learning_rate, optim,lr_warmup, scheduler, 
                                eta_min, patience, decay_rate=0.1):
    optimizer= _get_optimizer(model=model, learning_rate=learning_rate,optim=optim);
    if(optimizer is None):
        print("Wrong optimizer selected or some error is there");
        exit();
    
    scheduler= _get_scheduler(optim= optimizer,lr_warmup=lr_warmup, 
                             scheduler= scheduler, eta_min= eta_min, 
                              patience= patience, decay_rate= decay_rate);

    return optimizer,scheduler;


def get_head_sizes(n_heads,hidden_size, variable_size_heads):
    li=None
    if(variable_size_heads):
        # the size of all heads will be atleast temp;
        temp= hidden_size//n_heads;
        temp2= hidden_size//(4*n_heads);
        li=[temp-temp2]*n_heads;
        for i in range(0, n_heads,2):
            li[i]= temp+temp2;
    
    # Uncomment below line as and when needed
    return li;

def get_head_sizes_v2(n_heads,hidden_size, variable_size_heads):
    li=None
    if(variable_size_heads):
        # the size of all heads will be atleast temp;
        temp= hidden_size//n_heads;
        temp2= hidden_size//(4*n_heads);
        li=[temp]*n_heads;
        for i in range(n_heads//4):
            li[i]= temp-temp2;
        for i in range(n_heads//4,n_heads//2):
            li[i] = temp+temp2;

    return li;


