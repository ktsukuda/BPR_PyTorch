import configparser

import data


def main():
    config = configparser.ConfigParser()
    config.read('BPR_PyTorch/config.ini')

    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data('validation')
    test_data = data_splitter.make_evaluation_data('test')


if __name__ == "__main__":
    main()
