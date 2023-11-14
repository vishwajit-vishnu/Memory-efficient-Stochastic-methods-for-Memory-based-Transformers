
# modelsv2.py

# created by: Vishwajit kumar vishnu

# Models.py contains the main model details


import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np;
import math;
import copy;
from functools import partial; 
from adaptive_io import build_adaptive_io, compute_aux_loss ;

from visualisation import save_matrix;
########################
temp_models_count=0;
########################


# Size_notations:
# B= batch_size, H= hidden_size, M= block_size, L= mem_len(memory), E= EMbedding_size


# Refer to section 3.3 of the transformer-XL paper
# This will not be learnt
class PositionalEmbedding(nn.Module):   
    def __init__(self, hidden_size):
        super(PositionalEmbedding,self).__init__()
        self.hidden_size = hidden_size

        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size));

        #Using register_buffer will help skip the updation of 'self.pe' during backprop's step()
        #Also using register_buffer will not  count "self.pe" as a model parameter
        # Also it saves them to state_dict
        # For details : https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch#:~:text=Buffers%20are%20named%20tensors%20that,buffers%20will%20go%20as%20well.
      
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[ None,:,:] ; #  1*relative_sequence*H


def _create_decoder_mask(block_size, mem_len, same_length=False):
    temp=torch.tensor(());
    all_ones_matrix= temp.new_ones(block_size, block_size+mem_len);
    dec_mask= torch.triu(all_ones_matrix, diagonal= 1+mem_len).bool()#byte(); # Using byte() is deprecated

    if(same_length):
        qlen =block_size;
        klen= qlen + mem_len
        mask_len = klen - mem_len
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        dec_attn_mask = ( torch.triu(all_ones_matrix, diagonal= 1+mem_len) + 
                         torch.tril(all_ones_matrix, -mask_shift_len)).bool()#byte();
        return dec_attn_mask[None,:,:];

    # Mask is 1 at places we need not attend in the attention
    return dec_mask[None,:,:];   # 1*M*(L+M)



class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, mem_len, dropatt, dropout,
                 variable_size_heads, head_sizes_list, learn_bias_scales=False,
                 cross_head_attn= False, ch_prob=0.0, learn_head_importance= False):
        super(MultiHeadAttention,self).__init__();

        self.variable_size_heads= variable_size_heads;
        self.n_heads= n_heads;
        self.hidden_size= hidden_size;
        self.mem_len= mem_len;
        self.dropatt= nn.Dropout(dropatt)
        self.dropout= nn.Dropout(dropout) ;

        self.cross_head_attn= cross_head_attn;              # crosshead attn to diversify the outputs of each layer
        self.ch_prob= ch_prob;

        ######################################################################
        self.learn_head_importance= learn_head_importance;  # Head_importances are denoted as zeta_i for ith head;
        self.zeta= nn.ParameterList();
        self.learn_bias_scales=learn_bias_scales;   # Each head scales the biases using a bias_parameter for each of content_bias
                                                    # and positional_bias if the learn_bias_scales is True

        ########################################################################
        
        if(variable_size_heads):
            self.scale= [];
            self.w_q, self.w_k, self.w_v = nn.ModuleList(), nn.ModuleList(), nn.ModuleList() ;
            self.r_net= nn.ModuleList();
            for head_dim in (head_sizes_list):
                self.r_net.append(nn.Linear(hidden_size, head_dim, bias= False));
                self.w_q.append( nn.Linear(hidden_size, head_dim,bias=False) );
                self.w_k.append( nn.Linear(hidden_size, head_dim,bias=False) );
                self.w_v.append( nn.Linear(hidden_size, head_dim,bias=False) );
                tempval= 1/(head_dim**0.5);
                self.scale.append(tempval);
                if(self.learn_bias_scales):
                    self.content_bias_scales.append(nn.Parameter(torch.ones(1)));
                    self.positional_bias_scales.append(nn.Parameter(torch.ones(1)));
                if(self.learn_head_importance):
                    self.zeta.append(nn.Parameter(torch.zeros(1)));

        else:
            self.r_net= nn.Linear(hidden_size, hidden_size, bias= False) ;
            assert hidden_size % n_heads==0 , "n_heads do not divide hidden size"
            self.head_dim= self.hidden_size // self.n_heads;
            self.w_q= nn.Linear(hidden_size, hidden_size, bias= False) ;
            self.w_k = nn.Linear( hidden_size, hidden_size, bias= False);
            self.w_v = nn.Linear( hidden_size, hidden_size, bias= False);
            self.scale= 1/(self.head_dim**0.5);
            if(self.learn_bias_scales):
                self.content_bias_scales=nn.Parameter(torch.ones(self.n_heads));
                self.positional_bias_scales= nn.Parameter(torch.ones(self.n_heads));
            if(self.learn_head_importance):
                self.zeta = nn.Parameter(torch.ones(self.n_heads));

        ###################################################################
        self.w_o= nn.Linear(hidden_size, hidden_size, bias= False);


    #below function will do Q.K_transpose and then scale   
    def dot_product(self,Q,K,scale_value=1.0):
        # (Query:) B*n_heads* M * head_dim ; (Key.T :) B*n_heads* head_dim * (L+M)
        scores= torch.matmul(Q, torch.transpose(K,-1,-2));
        scores.mul_(scale_value);
        return scores;

    def calculate_attn(self, attn_score,V, block_size, mode=None, same_length= False):
        decoder_mask= _create_decoder_mask(block_size, self.mem_len, same_length= same_length);  # M*(L+M)
        decoder_mask= decoder_mask.to(V.device);
        attn_score=attn_score.masked_fill_(decoder_mask, -float('inf'));
        attn_probs= F.softmax(attn_score, dim=-1);

        ##########################################################
        global temp_models_count;
        #print('inside models, Q.shape',Q.shape,K.shape, V.shape);
        temp_models_count+=1; #print('temp_models_count',temp_models_count);
        if(mode=='test' and torch.distributed.get_rank()==0 and temp_models_count<1200):
            #print('saving the matrix');
            save_matrix(attn_matrix= attn_probs, count= str(temp_models_count));
        ##########################################################

        attn_probs= self.dropatt(attn_probs);
        return torch.matmul(attn_probs,V);

    def _do_cross_attn(self, K,R,V, key, pos_emb, value, query_dim=64):
        temp_prob= torch.rand(1).to(V.device);
        torch.distributed.broadcast(temp_prob,src=0);   #print(temp_random_no);
        if(temp_prob < self.ch_prob):
            if(self.variable_size_heads):
                random_head_no= torch.floor((torch.rand(1)*self.n_heads).to(key.device));
                torch.distributed.broadcast(random_head_no, src=0);
                while(self.w_k[int(random_head_no)].shape[-1] != query_dim):
                    random_head_no= torch.floor((torch.rand(1)*self.n_heads).to(key.device));
                    torch.distributed.broadcast(random_head_no, src=0);
                K= self.w_k[int(random_head_no)](key);
                R= self.r_net[int(random_head_no)](pos_emb);
                V= self.w_v[int(random_head_no)](value);
            else:
                # K.shape= B*n_heads*key_size*H
                shuffled_positions= torch.randperm(self.n_heads).to(key.device);
                torch.distributed.broadcast(shuffled_positions, src=0);
                K= K[:,shuffled_positions,:,:]
                R= R[:,shuffled_positions,:,:];                
                V=  V[:,shuffled_positions,:,:];

        return K, R, V;


    def forward(self, query, key, value, pos_emb, content_bias, positional_bias, 
                same_length= False, mode= None, prune_scale= None):

        # query= B *M (block size) *H
        batch_sz, M = query.size(0), query.size(1) ;
        temp_list= torch.tensor(()).to(query.device);

        ############################################################################
        if(self.variable_size_heads):
            for i, (wq, wk,wv, r_net) in enumerate(zip(self.w_q, self.w_k, self.w_v, self.r_net)):
                # wq(query) is of sim B * M* head_di
                Q, K, V, R= wq(query), wk(key), wv(value), r_net(pos_emb);

                if(self.cross_head_attn and mode=='train' ):
                    K, R, V= self._do_cross_attn(K= K, R=R, V=V, key=key, pos_emb= pos_emb, 
                                                  value= value, query_dim= Q.shape[-1] );
                
                R= R[None,:,:]   # 1* (L+M) * head_dim
                if(self.learn_bias_scales):
                    content_bias[i]= content_bias[i].mul(self.content_bias_scales[i].clamp(min=0.0, max=1.0));
                    positional_bias[i]= positional_bias[i].mul(self.positional_bias_scales[i].clamp(min=0.0, max=1.0));

                A_C= self.dot_product(Q+ content_bias[i][None,None,:], K= K,scale_value= self.scale[i]);
                B_D= self.dot_product(Q+ positional_bias[i][None,None,:], K= R,scale_value= self.scale[i]);
                
                attn_score= A_C + B_D
                out= self.calculate_attn(attn_score= attn_score,V= V, block_size=M,mode= mode, same_length= same_length)
                if(self.learn_head_importance):
                    out.mul_(self.zeta[i].clamp(min=0.0, max=1.0));
                if(prune_scale is not None):
                    out.mul_(prune_scale[None,i,None,None]);
                temp_list= torch.cat([temp_list,out],dim =-1);    # Since H(hidden_size) is last dimension to we do this
            
            assert temp_list.size(-1)== self.hidden_size, "the projected dimensions do not add to the hidden_size" ;
            out= temp_list.contiguous().view(batch_sz,-1,self.hidden_size);


        else:
            head_dim= self.hidden_size//self.n_heads;
            # reshape them to (batch_size, block size,n_heads, head_dim)
            # then permute to get the shape (batch_size,n_heads,block_size, head_dim)
            Q, K, V, R = self.w_q(query) , self.w_k(key) , self.w_v(value), self.r_net(pos_emb) ;

            # There are four terms in the transformer-XL paper
            # (a) content-based addressing: (EWq)WkE, (b) content-dependent positional bias (EWq)(WkR) (c)global content bias uWkE
            #  (d) global positional bias.v(WkR)

            Q= Q.reshape(batch_sz, M, self.n_heads, head_dim).permute(0,2,1,3);
            K= K.reshape(batch_sz,self.mem_len+M, self.n_heads, head_dim).permute(0,2,1,3) ;
            V= V.reshape(batch_sz,self.mem_len+M, self.n_heads, head_dim).permute(0,2,1,3) ;
            R= R.reshape(1,self.mem_len+M, self.n_heads, head_dim).permute(0,2,1,3) ;    # R before permute 1* (L+M) * n_heads * head_dim

            if(self.cross_head_attn and mode=='train' ):
                K, R, V= self._do_cross_attn(K= K,R=R, V=V, key=key,pos_emb= pos_emb, value=value);

            if(self.learn_bias_scales):
                content_bias= content_bias.mul(self.content_bias_scales.clamp(min=0.0, max=1.0)[:,None]);
                positional_bias= positional_bias.mul(self.positional_bias_scales.clamp(min=0.0, max=1.0)[:, None]);

                
            # since positional_bias and content_bias are of shape= n_heads* head_dim
            # we need to add the dimensions to do the addition without error
            A_C= self.dot_product(Q= Q+ content_bias[None,:,None,:], K= K,scale_value= self.scale);
            B_D= self.dot_product(Q= Q+ positional_bias[None,:,None,:],K= R,scale_value= self.scale);
            
            attn_score= A_C + B_D
            out= self.calculate_attn(attn_score= attn_score,V= V, block_size=M,mode= mode, same_length= same_length)
            if(self.learn_head_importance):
                out.mul_(self.zeta.clamp(min=0.0, max=1.0)[None,:,None,None]);

            if(prune_scale is not None):
                out.mul_(prune_scale[None,:,None,None]);

            out= out.permute(0,2,1,3).contiguous().view(batch_sz,-1,self.n_heads*head_dim);
            
        #######################################################################
        out= self.w_o(out); # B* M* H
        return self.dropout(out);



# SEE WHY TO USE NL_ff_input  and inwhich order to use in next text cell of this notebook
class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size,dropout=0.0, NL_ff_input= False):
        super(FeedForwardLayer,self).__init__();
        self.fc1= nn.Linear(hidden_size, inner_hidden_size);
        self.fc2= nn.Linear(inner_hidden_size, hidden_size);
        self.drop1= nn.Dropout(dropout);
        self.drop2= nn.Dropout(dropout);
        self.NL_ff_input = NL_ff_input;  # Nonlinear ff_input

    def _non_linearity(self, h,activation_fn):
        if(activation_fn is None or activation_fn=='relu'):
            h1= F.relu(h) ;
        elif(activation_fn== 'gelu'):
            h1= F.gelu(h) ;
        elif(activation_fn== 'selu'):
            h1= F.selu(h) ;
        else:
            print("unknown activation function chosen.. Exiting");
            exit();
        return h1;

    def forward(self,x, activation_fn= None):
        if(self.NL_ff_input):
            x= self._non_linearity(h=x,activation_fn= activation_fn);
            
        h= self.fc1(x);
        h1= self._non_linearity(h= h,activation_fn= activation_fn);
        h1= self.drop1(h1);
        h2= self.drop2(self.fc2(h1));
        return h2;


class TransformerLayer(nn.Module):
    def __init__(self,hidden_size, n_heads, mem_len, inner_hidden_size,dropout,activation_fn, 
                 dropatt, variable_size_heads, head_sizes_list,learn_bias_scales=False, 
                 same_length= False, skip_retain= False, prenorm= False, NL_ff_input= False,
                 cross_head_attn= False, ch_prob=0.0, learn_head_importance= False, random_relpos= False):
        super(TransformerLayer,self).__init__();

        self.attn= MultiHeadAttention( hidden_size= hidden_size, n_heads=n_heads, mem_len= mem_len, dropatt= dropatt,
                          dropout=dropout,variable_size_heads= variable_size_heads, head_sizes_list= head_sizes_list,
                          learn_bias_scales= learn_bias_scales, cross_head_attn= cross_head_attn, ch_prob= ch_prob, 
                          learn_head_importance= learn_head_importance) ;

        self.norm1= nn.LayerNorm(hidden_size) ;
        self.ff =FeedForwardLayer( hidden_size, inner_hidden_size,dropout, NL_ff_input) ;
        self.norm2= nn.LayerNorm(hidden_size);
        self.activation_fn= activation_fn;
        self.prenorm= prenorm;
        self.skip_retain= skip_retain;
        self.mem_len= mem_len;
        self.random_relpos= random_relpos ; 
        self.pos_encoder= PositionalEmbedding(hidden_size);

    def forward(self, h, mem, pos_emb, content_bias, positional_bias, current_block_position=1, 
                mem_block_position=0, mode= None, same_length= False, prune_scale= None):
        # h= B*M*H
        # mem= B*L(mem_len) *H

        _, block_sz, _= h.size();

        if(self.prenorm):
            h= self.norm1(h);

        h_all= torch.cat([mem, h], dim=1) ;  # B*(L+M)* H

        # If skip retain mechanism is used then we need to recalculate the relative memory positions using following way
        if(self.skip_retain and mode=='train'):
            #find howw to find for the memory posseq
            #print('Transformerlayer: MB', mem_block_position, 'CB', current_block_position);
            start_mem_pos= (current_block_position- mem_block_position)*block_sz+ self.mem_len
            end_mem_pos= (current_block_position- mem_block_position)*block_sz;
            pos_seq_mem= torch.arange(( start_mem_pos -1),(end_mem_pos-1), -1.0, device= h.device, dtype= h.dtype);
            pos_seq_query= torch.arange(( block_sz -1),-1, -1.0, device= h.device, dtype= h.dtype);
            if(self.random_relpos):
                random_shift= torch.randint(low= 0, high= 5000, size=(1,) , device= h.device, dtype= h.dtype) ;
                torch.distributed.broadcast(random_shift, src=0);
                pos_seq_mem, pos_seq_query = pos_seq_mem + random_shift, pos_seq_query + random_shift ;
            pos_seq= torch.cat([pos_seq_mem,pos_seq_query], dim=-1)
            pos_emb= self.pos_encoder(pos_seq);

        elif(self.random_relpos and mode=='train'):
            pos_seq= torch.arange(( block_sz+ self.mem_len -1),-1, -1.0, device= h.device, dtype= h.dtype);
            random_shift= torch.randint(low= 0, high= 5000, size=(1,) , device= h.device, dtype= h.dtype) ;
            torch.distributed.broadcast(random_shift, src=0);
            pos_seq = pos_seq + random_shift ;
            pos_emb= self.pos_encoder(pos_seq);
        

        attn_out= self.attn(query= h, key= h_all, value= h_all, pos_emb= pos_emb, content_bias= content_bias, 
                            positional_bias=positional_bias, same_length= same_length, mode= mode, 
                            prune_scale= prune_scale);
        h= h +attn_out ;
        
        if(self.prenorm):
            h= self.norm2(h) ;
        else:
            h= self.norm1(h);

        ff_out= self.ff(h, activation_fn= self.activation_fn);
        h= h+ ff_out;

        if( not self.prenorm):
            h = self.norm2(h) ;
        return h; 




def _rotate(self, vec):
    min, max = -math.pi , math.pi ;
    vec_dim= vec.size(-1);
    rotation_angle= (max- min) *torch.rand(1).to(vec.device) + min ; # generate rotation in  range(-pi,pi);
    positions=torch.tensor(())
    positions= torch.randint(low=0, high=vec_dim, size=(2,)).to(vec.device);

    torch.distributed.broadcast(rotation_angle,src=0);
    torch.distributed.broadcast(positions,src=0);

    rot_matrix= torch.eye(vec_dim).to(vec.device)[None,:,:];  
    rot_matrix[:, int(positions[0]), int(positions[0])]= torch.cos(rotation_angle);
    rot_matrix[:, int(positions[0]), int(positions[1])]= -torch.sin(rotation_angle);
    rot_matrix[:, int(positions[1]), int(positions[0])]= torch.sin(rotation_angle);
    rot_matrix[:, int(positions[1]), int(positions[1])]= torch.cos(rotation_angle);
    rotated_op= torch.matmul(rot_matrix,torch.transpose(vec, -1,-2)); # B*H*M
    return torch.transpose(rotated_op,-1,-2); #B*M*H

def _get_noise(self, tgt_len, hidden_size, noise_std, device):
    return noise_std*torch.randn(tgt_len, hidden_size).to(device)[None, :,:];   # 1*M*H


###   3. Layernorm vs scalenorm( already tested on transformers) vs switchableNorm or switchable-shake norm
###   ToDO: Paper for switchable shake http://www.cs.toronto.edu/~sajadn/sajad_norouzi/CSC2516.pdf
###   TODO While optimising remember to check the running time and flops for using each of the methods

class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, n_heads, n_layer, mem_len,inner_hidden_size,
                 emb_dropout, dropout, activation_fn, dropatt,adapt_io_cutoffs, adapt_io_divval, adapt_io_tied,
                 variable_size_heads, head_sizes_list, clamp_len=-1, same_length= False,adapt_io= False,
                 skip_retain=False, uniform_skip_prob=0.0, factorized_embedding= False, NL_ff_input= False,
                 learn_bias_scales= False, prenorm= False, share_weight= False, layer_chunk_size=1, 
                 cross_head_attn= False, ch_prob=0.0, learn_head_importance= False, rotation= False, 
                 random_relpos= False):

        super(Transformer,self).__init__() ;
        self.hidden_size= hidden_size;
        self.embedding_size= embedding_size ; 
        self.mem_len= mem_len;
        self.n_heads= n_heads;
        self.n_layer= n_layer;
        self.head_sizes_list= head_sizes_list;
        self.variable_size_heads= variable_size_heads;
        self.adapt_io= adapt_io;
        self.clamp_len=clamp_len;
        self.same_length= same_length;
        self.emb_dropout= nn.Dropout(emb_dropout);

        ################### Transformations  ########################
        self.rotation= rotation;

        ##############################################################################################
        self.skip_retain= skip_retain;
        self.current_block_position=0;
        self.mem_block_position= [0]* self.n_layer;
        if(skip_retain):
            # initially the block position of inputs is 1 and that of memory is 0.
            # EIther we will use uniform skipping in all layers
            self.uniform_skip_prob= uniform_skip_prob;
            # Functional skipping, the first and last layer will never be skipped
            self.functional_skip_prob=[];
            for i in range(self.n_layer):
                temp_prob= 0.5*i/n_layer
                self.functional_skip_prob.append(temp_prob);
            self.functional_skip_prob[self.n_layer-1]= 0.0;

        ######################    EMBEDDINGs    #############################
        self.factorized_embedding= factorized_embedding;
        # Use of factorized embedding parameters, similar to ALBERT
        if(factorized_embedding):
            self.factorized_inputs= nn.Linear(embedding_size, hidden_size);
            self.factorized_outputs= nn.Linear(hidden_size,embedding_size);
        else:
            self.embedding_size= embedding_size= self.hidden_size;

        if(self.adapt_io):
            self.in_emb , self.out_emb = build_adaptive_io(vocab_size, embedding_size, adapt_io_cutoffs,
                                                       adapt_io_divval, adapt_io_tied)
        else:
            self.in_emb= nn.Embedding(vocab_size, embedding_size);
            self.out_emb= nn.Linear(embedding_size,vocab_size);

        ############# #################################### ############
        self.pos_emb= PositionalEmbedding(hidden_size);

        if(variable_size_heads):
            #print("Using variable size heads");
            self.positional_bias= nn.ParameterList();
            self.content_bias= nn.ParameterList();
            for i in head_sizes_list:
                self.positional_bias.append(nn.Parameter(torch.randn(i)));
                self.content_bias.append(nn.Parameter(torch.randn(i)));
        else:
            assert hidden_size % n_heads==0, "Hidden size not divisible by the n_heads"
            head_dim= hidden_size// n_heads;
            self.positional_bias= nn.Parameter(torch.randn(n_heads, head_dim));
            self.content_bias= nn.Parameter(torch.randn(n_heads, head_dim));

        ##########################   several layers    #############################
        self.layers= nn.ModuleList();
        self.share_weight= share_weight;
        # We do not share weights with the first and last layers of the model
        # Its similar to subformer but we further use chunks of shared weight layers
        if(share_weight):
            self.layer_chunk_size= layer_chunk_size;
            self.shared_layers_list= self._get_shared_layers_list();
            n_layer= len(self.shared_layers_list);

        self.layers.extend(
            TransformerLayer(hidden_size= hidden_size, n_heads= n_heads, mem_len= mem_len, inner_hidden_size= inner_hidden_size,
                             dropout= dropout,activation_fn= activation_fn, dropatt= dropatt, 
                             variable_size_heads= variable_size_heads, head_sizes_list= head_sizes_list, 
                             learn_bias_scales= learn_bias_scales,same_length= self.same_length, 
                             skip_retain= self.skip_retain, prenorm= prenorm, NL_ff_input= NL_ff_input,
                             cross_head_attn= cross_head_attn, ch_prob= ch_prob, learn_head_importance= learn_head_importance, 
                             random_relpos= random_relpos )
            for i in range(n_layer)
        ) ;
        #########################     ############################
    
    def reset_positions(self):
        min_pos= min(self.mem_block_position);
        self.mem_block_position= [i-min_pos for i in self.mem_block_position];
        self.current_block_position= self.current_block_position- min_pos;

    # If following function outputs [0,3,6,9,10,11] it means layers 1,2,3 share weights
    # and layers (3+1) till 6(including 6) share weight and so on , and layers 0, 10, 11 do not share weight for a 12 layer model
    def _get_shared_layers_list(self):
        n_layer, layer_chunk_size = self.n_layer, self.layer_chunk_size;
        assert layer_chunk_size < n_layer, 'Layer_chunk size for weight sharing bigger than n_layer'
        shared_layers_list=[];
        i=0;
        while(i< n_layer-1):
            shared_layers_list.append(i);
            i=i+layer_chunk_size;

        i=i- layer_chunk_size;
        while(i>0 and i<n_layer-2):
            i+=1;
            shared_layers_list.append(i)
        shared_layers_list.append(n_layer-1)
        print('shared_layers_list',shared_layers_list);
        return shared_layers_list

    def _get_shared_layer_index(self,l_no):
        for i in range(len(self.shared_layers_list)):
            if(l_no<= self.shared_layers_list[i]):
                return i;

    def _skip_this_layer(self, skip_prob, device):
        #generate a random no between 0 and 1
        temp_random_no= torch.rand(1).to(device);
        torch.distributed.broadcast(temp_random_no,src=0);  
        if(temp_random_no < skip_prob):
            return True;
        else:
            return False;


    def forward(self,x, target,mem, mode= None, prune_elements= None, train_mode= None):
        self.current_block_position+=1;
        device= x.device;
         
        ################### Input embedding     ####################################
        # x is input of size Batch_size * Block_size = (B*M)
        batch_sz, block_sz= x.size();
        h= self.in_emb(x) ;     # B*M*E
        if(self.factorized_embedding):        # this means E != H
            h= self.factorized_inputs(h);      # B*M*H
        h- self.emb_dropout(h);
        

        if(self.rotation and mode =='train'):
            self._rotate(h);

        ##################    Positional Encoding      ##########################
        pos_seq= torch.arange(( block_sz+ self.mem_len -1),-1, -1.0, device= h.device, dtype= h.dtype);
        
        if(self.clamp_len>0):
            # The relative positions are like 5,4,3,2,1,0. So if clamp_len=2 then it becomes: 2,2,2,2,1,0
            pos_seq.clamp_(max=self.clamp_len);
        pos_emb= self.pos_emb(pos_seq);

        mem_next= copy.deepcopy(mem);

        ###################  going through the layers    #################
        l=0;
        while (l< self.n_layer):

            ###########################################################
            if(self.skip_retain and mode =='train' and not(train_mode=='finetune')):
                self.reset_positions();
                if (self.uniform_skip_prob>0):
                    skip_prob= self.uniform_skip_prob;
                else:
                      skip_prob= self.functional_skip_prob[l];
                if(self._skip_this_layer(skip_prob= skip_prob, device=device)):
                    l+=1;
                    continue;
            elif(self.skip_retain and mode=='train' and train_mode=='freeze_pos'):
                pass;      # we do not change the mem_block_position and current_block_position
            else:
                self.mem_block_position= [0]* self.n_layer;
                self.current_block_position=1;
            ############################################

            if(self.mem_len > block_sz):
                mem_next_l= torch.cat([mem[l][:,- self.mem_len+ block_sz: ,:], h], dim=-2).detach();
            else:
                mem_next_l=  h[:,-self.mem_len:,:].detach() ;
            mem_next[l]= mem_next_l ;

            if(self.share_weight):
                templ= self._get_shared_layer_index(l_no= l);
            else:
                templ =l;

            # ---------------------------------------------
            if(prune_elements is not None):
                if(templ== prune_elements[0]):
                    prune_scale= prune_elements[1];
                else:
                    prune_scale= None;
            else:
                prune_scale= None;
            # --------------------------------------------

            h= self.layers[templ] ( h=h, mem= mem[templ], pos_emb= pos_emb, content_bias= self.content_bias,
                   positional_bias= self.positional_bias, current_block_position= self.current_block_position,
                    mem_block_position= self.mem_block_position[templ],mode= mode, 
                    same_length= self.same_length, prune_scale= prune_scale ); # B*M*H
            ########################################## 
            ########################################## 
            l+=1 ;
            self.mem_block_position[templ]+=1;

        ####################################################################
        
        #print('MB', self.mem_block_position, 'CB', self.current_block_position, 'device',device);

        h= self.emb_dropout(h);
        if(self.factorized_embedding):
            h= self.factorized_outputs(h);

        if self.adapt_io:
            out = self.out_emb(h, target, mode=mode)
            ############################################
            if(mode=='generate'):
                return out;
            ###########################################
            original_loss = out.mean() 
            aux_loss= compute_aux_loss(in_emb= self.in_emb, out_emb= self.out_emb)
        else:
            out = F.log_softmax(self.out_emb(h), dim=-1)
            out = out.view(-1, out.size(-1))
            ############################################
            if(mode=='generate'):
                return out;
            ###########################################
            original_loss = F.nll_loss(out, target.view(-1));
            aux_loss= torch.zeros(1).to(original_loss.device)

        return original_loss, mem_next, aux_loss ;

############ models.py done ###################

