import os
import json
import tqdm
import torch
import configparser
from torch import nn, optim

import data
import evaluation
from BPR import BPR


def train(model, opt, data_splitter, validation_data, batch_size, config):
    epoch_data = []
    best_model_state_dict = None
    best_ndcg = 0
    for epoch in range(config.getint('MODEL', 'epoch')):
        model.train()
        train_loader = data_splitter.make_train_loader(config.getint('MODEL', 'n_negative'), batch_size)
        total_loss = 0
        for batch in tqdm.tqdm(train_loader):
            users, pos_items, neg_items = batch[0], batch[1], batch[2]
            users = users.to('cuda:0')
            pos_items = pos_items.to('cuda:0')
            neg_items = neg_items.to('cuda:0')
            opt.zero_grad()
            pos_pred = model(users, pos_items)
            neg_pred = model(users, neg_items)
            loss = - (pos_pred - neg_pred).sigmoid().log().sum()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        hit_ratio, ndcg = evaluation.evaluate(model, validation_data, config.getint('EVALUATION', 'top_k'))
        epoch_data.append({'epoch': epoch, 'loss': total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
        if ndcg > best_ndcg:
            best_model_state_dict = model.state_dict()
        print('[Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
    return epoch_data, best_model_state_dict


def save_train_result(best_model_state_dict, epoch_data, batch_size, lr, latent_dim, l2_reg, config):
    result_dir = "data/train_result/batch_size_{}-lr_{}-latent_dim_{}-l2_reg_{}-epoch_{}-n_negative_{}-top_k_{}".format(
        batch_size, lr, latent_dim, l2_reg, config['MODEL']['epoch'], config['MODEL']['n_negative'], config['EVALUATION']['top_k'])
    os.makedirs(result_dir, exist_ok=True)
    torch.save(best_model_state_dict, os.path.join(result_dir, 'model.pth'))
    with open(os.path.join(result_dir, 'epoch_data.json'), 'w') as f:
        json.dump(epoch_data, f, indent=4)


def find_best_model(config, n_user, n_item):
    best_model = None
    best_params = {}
    best_ndcg = 0
    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
                for l2_reg in map(float, config['MODEL']['l2_reg'].split()):
                    result_dir = "data/train_result/batch_size_{}-lr_{}-latent_dim_{}-l2_reg_{}-epoch_{}-n_negative_{}-top_k_{}".format(
                        batch_size, lr, latent_dim, l2_reg, config['MODEL']['epoch'], config['MODEL']['n_negative'], config['EVALUATION']['top_k'])
                    with open(os.path.join(result_dir, 'epoch_data.json')) as f:
                        ndcg = max([d['NDCG'] for d in json.load(f)])
                        if ndcg > best_ndcg:
                            best_ndcg = ndcg
                            best_params = {
                                'batch_size': batch_size, 'lr': lr, 'latent_dim': latent_dim, 'l2_reg': l2_reg}
                            model = BPR(n_user, n_item, latent_dim)
                            model.to('cuda:0')
                            model.load_state_dict(torch.load(os.path.join(result_dir, 'model.pth')))
                            best_model = model
    return best_model, best_params


def main():
    config = configparser.ConfigParser()
    config.read('BPR_PyTorch/config.ini')

    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data('validation')
    test_data = data_splitter.make_evaluation_data('test')

    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
                for l2_reg in map(float, config['MODEL']['l2_reg'].split()):
                    print('batch_size = {}, lr = {}, latent_dim = {}, l2_reg = {}'.format(
                        batch_size, lr, latent_dim, l2_reg))
                    model = BPR(data_splitter.n_user, data_splitter.n_item, latent_dim)
                    model.to('cuda:0')

                    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
                    epoch_data, best_model_state_dict = train(model, opt, data_splitter, validation_data, batch_size, config)
                    save_train_result(best_model_state_dict, epoch_data, batch_size, lr, latent_dim, l2_reg, config)

    best_model, best_params = find_best_model(config, data_splitter.n_user, data_splitter.n_item)
    hit_ratio, ndcg = evaluation.evaluate(best_model, test_data, config.getint('EVALUATION', 'top_k'))
    print('---------------------------------\nBest result')
    print('batch_size = {}, lr = {}, latent_dim = {}, l2_reg = {}'.format(
        best_params['batch_size'], best_params['lr'], best_params['latent_dim'], best_params['l2_reg']))
    print('HR = {:.4f}, NDCG = {:.4f}'.format(hit_ratio, ndcg))


if __name__ == "__main__":
    main()
