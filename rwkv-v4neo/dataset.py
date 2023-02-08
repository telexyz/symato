########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        if args.data_type == "symato":
            import sys; sys.path.append('../')
            from symato import Symato
            smt = Symato()
            self.data = smt.tokenize(args.data_file, rev=(args.data_order == "reversed"))
            print("\n\n- - - [ TRAIN DATA SAMPLE ] - - -\n", smt.tids_to_utf8(self.data[-800:]), "\n\n")
            self.vocab_size = smt.vocab_size()
            self.data_size = len(self.data)
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")

        elif args.data_type == "chars":
            self.data = open(args.data_file, "r", encoding="utf-8").read()
            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-8") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

        else: # unicode
            txt = open(args.data_file, "r", encoding=args.data_type).read()
            from tokenization_phobert_fast import PhobertTokenizerFast
            os.environ["TOKENIZERS_PARALLELISM"] = "False"
            tknz = PhobertTokenizerFast("./data/vocab.txt", "./data/bpe.codes", "./data/tokenizer.json")
            self.vocab_size = 64256 # 251 * 256
            self.data = tknz.encode(txt)
            self.data_size = len(self.data)
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
            rank_zero_info(f"Data has {self.data_size} samples.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        ctx_len = args.ctx_len # ctx_len là độ dài chuỗi token đầu vào
        req_len = ctx_len + 1  # cộng thêm một token là kết quả đầu ra 
        i = np.random.randint(0, self.data_size - req_len)
        if args.data_type == "chars": dix = [self.stoi[s] for s in self.data[i : i + req_len]]
        else: dix = self.data[i : i + req_len]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
