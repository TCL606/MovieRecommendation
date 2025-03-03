import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import argparse
import random
import numpy as np
import json
import time
import torch.nn.functional as F

class UVModel(nn.Module):
    def __init__(self, dim_user, dim_item, dim_hid):
        super(UVModel, self).__init__()
        self.U = nn.Parameter(torch.randn(dim_user, dim_hid), requires_grad=True)
        self.V = nn.Parameter(torch.randn(dim_item, dim_hid), requires_grad=True)

    def forward(self):
        return self.U @ self.V.t()

def train_model(model, X, mask, X_test, mask_test, lbd, lr, num_iters):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    res = []
    for it in tqdm(range(num_iters)):
        model.train()
        optimizer.zero_grad()

        pred = model()
        loss = 0.5 * torch.norm(mask * (X - pred), p='fro') ** 2 + lbd * (torch.norm(model.U, p='fro') ** 2 + torch.norm(model.V, p='fro') ** 2)

        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            print(f'Iter: {it}, Loss: {loss.item()}')
        
        rmse = test_model(model, X_test, mask_test)
        res.append((it, loss.item(), rmse.item()))

    return model, res

def test_model(model, X_test, mask_test):
    model.eval()
    pred = model()
    rmse = torch.sqrt(torch.sum((mask_test * (X_test - pred)) ** 2) / torch.sum(mask_test))
    return rmse

def preprocess(train_txt, test_txt, user_txt, dim_user, dim_item):
    user_map = {}
    tmp = 0
    with open(user_txt, 'r') as fp:
        for line in tqdm(fp):
            line = int(line.strip())
            user_map[line] = tmp
            tmp += 1

    X_train = torch.zeros([dim_user, dim_item], requires_grad=False)
    with open(train_txt, 'r') as fp:
        for line in tqdm(fp):
            line = line.strip().split()
            X_train[user_map[int(line[0])], int(line[1]) - 1] = int(line[2])
    mask_train = X_train != 0

    X_test = torch.zeros([dim_user, dim_item], requires_grad=False)
    with open(test_txt, 'r') as fp:
        for line in tqdm(fp):
            line = line.strip().split()
            X_test[user_map[int(line[0])], int(line[1]) - 1] = int(line[2])
    mask_test = X_test != 0
    return user_map, X_train, mask_train, X_test, mask_test

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def direct_pred(X, mask):
    X = mask * X
    mal_X = X @ X.t()
    norm_x = torch.norm(X, dim=1, keepdim=True)
    sim = mal_X / (norm_x @ norm_x.t())

    pred = (sim @ X) / (sim @ torch.ones(sim.shape))
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2')
    parser.add_argument('--train_txt', type=str, default="/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/lp_proj/Project2-data/netflix_train.txt")
    parser.add_argument('--text_txt', type=str, default="/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/lp_proj/Project2-data/netflix_test.txt")
    parser.add_argument('--user_txt', type=str, default="/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/lp_proj/Project2-data/users.txt")
    parser.add_argument('--dim_user', type=int, default=10000)
    parser.add_argument('--dim_item', type=int, default=10000)
    parser.add_argument('--output_root', type=str, default="/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/lp_proj/output")
    parser.add_argument('--output_name', type=str, default="debug")
    parser.add_argument('--dim_feat', type=int, default=50)
    parser.add_argument('--lbd', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--direct_test', action='store_true')
    
    args = parser.parse_args()

    set_seed(args.seed)

    train_txt = args.train_txt
    test_txt = args.text_txt
    user_txt = args.user_txt
    dim_user, dim_item = args.dim_user, args.dim_item
    dim_feat = args.dim_feat
    user_map, X_train, mask_train, X_test, mask_test = preprocess(train_txt, test_txt, user_txt, dim_user, dim_item)

    if args.direct_test:
        start_time = time.time()
        pred = direct_pred(X_train, mask_train)
        end_time = time.time()
        rmse = torch.sqrt(torch.sum((mask_test * (X_test - pred)) ** 2) / torch.sum(mask_test))

        print(f"RMSE: {rmse}, Time: {end_time - start_time}")
    else:
        if not args.do_test:
            output_dir = os.path.join(args.output_root, args.output_name)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
                json.dump(vars(args), fp, indent=4)


            lbd = args.lbd
            lr = args.lr
            num_iters = args.num_iters

            W, H = X_train.shape
            model = UVModel(W, H, dim_feat)
            if args.model is not None:
                ckpt = torch.load(args.model, weights_only=False)
                model.load_state_dict(ckpt)

            model = model.cuda()
            X_train = X_train.cuda()
            mask_train = mask_train.cuda()
            X_test = X_test.cuda()
            mask_test = mask_test.cuda()
                
            start_time = time.time()
            model, res = train_model(model, X_train, mask_train, X_test, mask_test, lbd, lr, num_iters)
            end_time = time.time()

            print(end_time - start_time)
            rmse = test_model(model, X_test, mask_test)
            with open(os.path.join(output_dir, 'time.json'), 'w') as fp:
                json.dump({"time": end_time - start_time, "rmse": rmse.item()}, fp, indent=4)
            
            print({"time": end_time - start_time, "rmse": rmse.item()})

            torch.save(model.state_dict(), os.path.join(output_dir, "model.bin"))

            plt.figure()
            plt.plot([it[0] for it in res], [it[1] for it in res])
            plt.savefig(os.path.join(output_dir, "loss.png"))

            plt.figure()
            plt.plot([it[0] for it in res], [it[2] for it in res])
            plt.savefig(os.path.join(output_dir, "rmse_test.png"))
        
        else:
            W, H = X_train.shape
            model = UVModel(W, H, dim_feat)
            if args.model is not None:
                ckpt = torch.load(args.model, weights_only=False)
                model.load_state_dict(ckpt)
            model = model.cuda()
            X_test = X_test.cuda()
            mask_test = mask_test.cuda()
            rmse = test_model(model, X_test, mask_test)
            breakpoint()
            print(rmse.item())
