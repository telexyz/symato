########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from binidx import MMapIndexedDataset
from utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "binidx":
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            if args.data_file.endswith('/'):
                d_all = [] # tìm trong dir các file .idx
                for p in os.listdir(args.data_file):
                    if p.endswith(".idx"): d_all += [p[:-4]]
                d_all.sort()
                rank_zero_info(d_all)
                exit(0)
            else:
                self.data = MMapIndexedDataset(args.data_file)
                self.data_size = len(self.data._bin_buffer) // 2
                rank_zero_info(f"Data has {self.data_size} tokens.")

            if args.my_pile_stage > 0:
                # assert self.data_size == 332115325534 and self.vocab_size == 50277
                self.samples_per_epoch = args.epoch_steps * args.real_bsz
                assert self.samples_per_epoch == 40320
                rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
                dataset_slot = self.data_size // args.ctx_len
                assert MaybeIsPrime(args.magic_prime)
                assert args.magic_prime % 3 == 2
                assert args.magic_prime / dataset_slot > 0.99 and args.magic_prime / dataset_slot <= 1

        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            rank_zero_info("Current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens.")

        elif args.data_type == "uint16":
            self.data = np.fromfile(args.data_file, dtype=np.uint16).astype("int32").reshape(-1, args.my_sample_len)
            self.vocab_size = args.vocab_size
            rank_zero_info("Current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = self.data.shape[0]
            rank_zero_info(f"Data has {self.data_size} samples.")

        else:
            if args.data_type == "dummy":
                rank_zero_info("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r", encoding=args.data_type).read()

            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-16le") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz


    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        if args.data_type == "uint16":
            i = np.random.randint(0, self.data_size-1)
            dix = self.data[i]
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)

        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            magic_prime = args.magic_prime
            data = self.data

            if args.my_pile_stage > 0:
                ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank
                factor = (math.sqrt(5) - 1) / 2
                factor = int(magic_prime * factor)
                i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                if data == self.data_pile:
                    i = i + args.my_pile_shift
                # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
            else:
                # cheat: pick a random spot in dataset
                i = np.random.randint(0, self.data_size - req_len)

            if args.data_type == "binidx":
                dix = data.get(idx=0, offset=i, length=req_len).astype(int)
            # 
            elif args.data_type == "numpy":
                dix = data[i : i + req_len]
            # 
            else:
                dix = [self.stoi[s] for s in data[i : i + req_len]]

            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
            
        return x, y
