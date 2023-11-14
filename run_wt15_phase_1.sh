#!/bin/bash



ngpus=2
args="" 
args="
--distributed \
--adapt_io \
--batch_size 32 \
--data ../gputest/D4_my_code/data/wikitext-103 \
--dataset wt103 \
--n_layer 15 \
--hidden_size 624 \
--inner_hidden_size 2496 \
--n_heads 16 \
--optim adam \
--lr 0.00025 \
--scheduler cosine \
--warmup 16000 \
--patience 3 \
--decay_rate 0.8 \
--max_steps 1200000 \
--tgt_len 512 \
--mem_len 512 \
--eval_tgt_len 128 \
--eval_batch_size 10 \
--grad_clip 0.25 \
--dropout 0.15 \
--dropatt 0.0 \
--emb_dropout 0.0 \
--work_dir wt15_final_corrected \
--activation_fn relu \
--log_interval 1000 \
--eval_interval 4000
"


echo "testing before train"
python3 -m torch.distributed.launch --nproc_per_node=$ngpus ../code_march_corrected/main.py $args \
  --mode test --eval_batch_size 10 --batch_size 10 --eval_tgt_len 128 --tgt_len 128  --mem_len 640 --clamp_len 800  --same_length

python3 -m torch.distributed.launch --nproc_per_node=$ngpus ../code_march_corrected/main.py $args \
  --mode test --eval_batch_size 10 --batch_size 10 --eval_tgt_len 128 --tgt_len 128  --mem_len 2000 --clamp_len 800  --same_length

echo "training ....."
python3 -m torch.distributed.launch --nproc_per_node=$ngpus ../code_march_corrected/main.py $args \
   --mode train  \
   --skip_retain  \
   --cross_head_attn --ch_prob 0.1   --dropout 0.15
