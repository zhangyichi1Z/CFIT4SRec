import os


sim = ['dot']
lmd = [0.1]
drop = [0.3]
tau = [1]
learning_rate = [0.001]
max_l = [50]
weight_decay_lst = [0, 0.001, 0.0001]
train_r = [0.25, 0.25, 0.5, 0.75, 1]
dataset = ["Amazon_Beauty","Amazon_Clothing_Shoes_and_Jewelry","Amazon_Sports_and_Outdoors","ml-1m"]


for l_ok in [0, 1]:
    for h_ok in [0, 1]:
        for b_ok in [0, 1]:
            for l in lmd:
                for s in sim:
                    for d in drop:
                        for data in dataset:
                            for t in tau:
                                for lr in learning_rate:
                                    for ml in max_l:
                                        os.system("python run_seq.py --dataset={} --train_batch_size=256 "
                                                  "--lmd={}  --model='CFIT4SRec'  --sim={} "
                                                  "--hidden_dropout_prob={} --attn_dropout_prob={} --tau={} --learning_rate={} "
                                                  "--MAX_ITEM_LIST_LENGTH={} --l_ok={} --h_ok={} --b_ok={}".format(
                                            data,
                                            l,
                                            s,
                                            d,
                                            d,
                                            t,
                                            lr,
                                            ml,
                                            l_ok,
                                            h_ok,
                                            b_ok
                                        ))
