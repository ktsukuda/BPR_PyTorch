import tqdm
import configparser
from torch import nn, optim

import data
import evaluation
from BPR import BPR


def train(model, opt, data_splitter, validation_data, batch_size, config):
    epoch_data = []
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
        print('[Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
    return epoch_data


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
                    epoch_data = train(model, opt, data_splitter, validation_data, batch_size, config)


if __name__ == "__main__":
    main()
