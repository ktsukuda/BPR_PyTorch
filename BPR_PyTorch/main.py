import configparser

import data
from BPR import BPR


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


if __name__ == "__main__":
    main()
