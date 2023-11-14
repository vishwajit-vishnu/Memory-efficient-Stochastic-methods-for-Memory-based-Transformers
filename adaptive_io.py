
# adaptive_io.py 

import torch;
import torch.nn as nn;
import torch.nn.functional as F;


#paper at : https://arxiv.org/abs/1809.10853
class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_tokens, d_embed, d_proj, cutoffs, div_val=4):
        super(AdaptiveEmbedding,self).__init__();
        self.n_tokens=n_tokens;
        self.d_embed= d_embed;
        self.d_proj= d_proj

        assert 0< min(cutoffs) <= max(cutoffs) < n_tokens , 'Cutoff ranges not correct';
        self.cutoff_ends= [0] + cutoffs + [n_tokens];

        #assert bellow statement as experiments show it and div_val=1 uses more memory
        assert div_val> 1, 'div_val should be greater than 1'
        self.div_val=div_val;
        self.emb_scale= d_proj ** 0.5;

        self.emb_layers= nn.ModuleList();
        self.emb_projs= nn.ParameterList();

        for i in range(len(self.cutoff_ends)-1):
            left_idx, right_idx = self.cutoff_ends[i] , self.cutoff_ends[i+1];
            d_emb_i= d_embed //(div_val **i);
            self.emb_layers.append(nn.Embedding(right_idx- left_idx, d_emb_i));
            self.emb_projs.append(nn.Parameter(torch.randn(d_proj, d_emb_i)));
         

    def forward(self, input):
        input_flat= input.contiguous().view(-1);
        emb_flat= torch.zeros((input_flat.size(0),self.d_proj), device=input.device);  # (BXM)*d_proj

        for i in range(len(self.cutoff_ends)-1):
            left_idx, right_idx= self.cutoff_ends[i], self.cutoff_ends[i+1] ;
            
            mask_i = (input_flat >= left_idx) & (input_flat< right_idx);
            indices_i= mask_i.nonzero(). squeeze();     # indices of the input values belonging to this range=[left_idx, right_idx)

            if(indices_i.numel()==0):
                continue;
            
            # indices_flat.index_select(0, indices_i) selects all the values at each of indices_i
            input_i= input_flat.index_select(0, indices_i) - left_idx;
            emb_i= self.emb_layers[i](input_i);          # emb_i shape: (no_of_values in this renge)*d_emb_i
            emb_i= F.linear(emb_i, self.emb_projs[i]);   # emb_i shape: (no_of_values in this renge)*d_proj

            emb_flat= emb_flat.type_as(emb_i);
            # index_copy documentation: https://pytorch.org/docs/stable/tensors.html
            emb_flat.index_copy_(0, indices_i, emb_i);   # # (BXM)*d_proj

        embeddings= emb_flat.view(*input.size(),self.d_proj);   #B*M*d_proj and d_proj= embedding_size(hidden_size)
        embeddings.mul_(self.emb_scale);

        return embeddings;


# paper available at http://arxiv.org/abs/1609.04309
class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_tokens, d_embed,d_proj, cutoffs,div_val):
        super(ProjectedAdaptiveLogSoftmax, self).__init__();
        self.n_tokesn= n_tokens;
        self.d_embed= d_embed;
        self.d_proj= d_proj;
        self.div_val=div_val;
        assert div_val> 1, 'div_val should be greater than 1';

        assert 0< min(cutoffs) <= max(cutoffs) < n_tokens , 'Cutoff ranges not correct';
        self.cutoff_ends= [0] + cutoffs +[n_tokens];

        self.shortlist_size= self.cutoff_ends[1];
        self.n_clusters= len(self.cutoff_ends)-2;
        self.head_size= self.shortlist_size + self.n_clusters;

        self.cluster_proj= nn.Linear(self.d_embed, self.n_clusters);

        self.out_layers= nn.ModuleList();
        self.out_projs = nn.ParameterList();

        for i in range(len(self.cutoff_ends)-1):
            left_idx, right_idx= self.cutoff_ends[i], self.cutoff_ends[i+1];
            d_emb_i= self.d_embed// (self.div_val**i);
            self.out_layers.append(nn.Linear(d_emb_i, right_idx-left_idx));
            self.out_projs.append(nn.Parameter(torch.randn(d_proj, d_emb_i)));

    def _compute_logit(self, hidden,weight, bias, proj):
        proj_hid= F.linear(hidden, proj.t().contiguous());
        logit= F.linear(input= proj_hid, weight= weight, bias= bias);
        return logit;

    def forward(self,hidden, target, mode= None):
        # hidden= B*M*H of Floattensor , target= B*M of Longtensor
        assert hidden.shape[-1] == self.d_proj,'Adapt_IO: hidden o/p dimension dont match d_proj' ;
        assert hidden.shape[:-1]== target.shape, 'Adapt_io: hidden shape and tensor shape dint match' ;

        target_shape= target.shape;
        hidden= hidden.contiguous().view(-1, self.d_proj);
        target= target.view(-1);

        weights, biases=[],[];
        for i in range(len(self.cutoff_ends)-1):
            left_idx, right_idx= self.cutoff_ends[i], self.cutoff_ends[i+1];

            weight_i= self.out_layers[i].weight;
            bias_i= self.out_layers[i].bias;

            if(i==0):
                weight_i= torch.cat([weight_i, self.cluster_proj.weight], dim=0);
                bias_i= torch.cat([bias_i, self.cluster_proj.bias], dim=0);
            
            weights.append(weight_i);
            biases.append(bias_i);

        # head weight and biases are at index=0
        head_logit= self._compute_logit(hidden, weights[0], biases[0],self.out_projs[0]);
        head_logprob= F.log_softmax(head_logit, dim=1);

        nll= torch.zeros_like(target, dtype= hidden.dtype, device= hidden.device);

        ####################################
        # TODO:
        generated_logprobs= [head_logprob[:, :self.cutoff_ends[1]]];
        ####################################

        offset=0;
        for i in range(len(self.cutoff_ends)-1):
            left_idx, right_idx= self.cutoff_ends[i], self.cutoff_ends[i+1];  #range= right_idx- left_idx
            mask_i = (target>= left_idx) & (target< right_idx);
            ##############################
            # TODO:
            if(mode=='generate'):
                mask_i= (target>0)
            ##############################
            indices_i= mask_i.nonzero().squeeze();
             
            if(indices_i.numel()==0):
              continue;
            
            target_i= target.index_select(0, indices_i) - left_idx;
            head_logprob_i=  head_logprob.index_select(0, indices_i);  # shape: (n_elements in head)*(head_size+n_clusters);
            
            if(i==0):
                logprob_i= head_logprob_i.gather(1, target_i[:, None]).squeeze(1); # shape: n_elements in head range
            else:
                hidden_i = hidden.index_select(0, indices_i);

                tail_logit_i= self._compute_logit(hidden_i, weights[i], biases[i], self.out_projs[i]);
                tail_logprob_i= F.log_softmax(tail_logit_i, dim=1);      #shape: (n_elements in this tail)* (right_idx-left_idx)

                #############################################
                if(mode=='generate'):
                    generated_logprobs.append(head_logprob_i[:,-i] + tail_logprob_i);
                #############################################

                logprob_i= head_logprob_i[:, -i] + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)


            nll.index_copy_(0, indices_i, -logprob_i);
            offset+= logprob_i.size(0);

        ########################################
        # TODO: NOt tested
        if( mode== 'generate'):
            print('generated_logprobs', generated_logprobs);
            generated_expprobs= torch.exp(torch.cat(generated_logprobs, dim=1));
            print('generated_expprobs shape:',generated_expprobs.size());
            print('generated_expprobs[0]', generated_expprobs[0]);
            return generated_expprobs;                    #TRY using F.Log_softmax instead of exp
        ############################################

        return nll.view(target_shape);   # shape: B*M


def _test_adaptive_softmax():
    ad= ProjectedAdaptiveLogSoftmax( n_tokens=20, d_embed=32,d_proj=32, cutoffs=[8],div_val=4);
    target= torch.tensor([[4,15,18],[14, 16, 1]])
    hidden= torch.randn(2,3,32)
    output= ad(hidden, target )
    print(output)

def _test_adaptive_embed():
    adem= AdaptiveEmbedding( n_tokens=20, d_embed=32, d_proj=32, cutoffs=[10], div_val=4);
    input= torch.tensor([[4,15,18],[14, 16, 1]]);
    out_emb= adem(input);
    print(out_emb.shape)

def compute_aux_loss(in_emb, out_emb):
    aux_loss= 0.0 *(
                    sum( x.weight[0,0] for x in in_emb.emb_layers) +
                    sum( x[0,0] for x in in_emb.emb_projs) +
                    sum( x[0,0] for x in out_emb.out_projs) +
                    sum( x.weight[0,0] for x in out_emb.out_layers) +
                    sum(x.bias[0] for x in out_emb.out_layers)
                    );
    return aux_loss;


def build_adaptive_io(vocab_size, embedding_size, adapt_io_cutoffs, adapt_io_divval, adapt_io_tied):
    in_emb= AdaptiveEmbedding(n_tokens= vocab_size, d_embed= embedding_size,d_proj= embedding_size, 
                              cutoffs= adapt_io_cutoffs,div_val= adapt_io_divval);
    out_emb= ProjectedAdaptiveLogSoftmax(n_tokens= vocab_size, d_embed= embedding_size,d_proj= embedding_size, 
                              cutoffs= adapt_io_cutoffs,div_val= adapt_io_divval);
    if(adapt_io_tied):
        for i in range(len(adapt_io_cutoffs)+1):
            out_emb.out_layers[i].weight= in_emb.emb_layers[i].weight ;
            out_emb.out_projs[i]= in_emb.emb_projs[i];

    return in_emb, out_emb;

############ adaptive_io.py done checking###################
