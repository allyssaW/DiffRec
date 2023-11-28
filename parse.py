import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--seed', type=int, default=2024,
                        help='random seed')
    parser.add_argument('--model', type=str, default='lgn',
                        help='rec-model, support [lgn]')
    parser.add_argument('--gpu', type=str, default=None,
                        help='training device')
    parser.add_argument('--log', type=str, default=None,
                        help='logging file path')

    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--hidden', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', action='store_true', default=False,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")

    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [gowalla, yelp2018, amazon-book]")
    parser.add_argument('--topk', type=int, default=20,
                        help="@k test list")
    parser.add_argument('--multicore', type=int, default=0,
                        help='whether to use multiprocessing or not in test')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether to save model weights')
    parser.add_argument('--load', action='store_true', default=False,
                        help='whether to load from stored weights')
    return parser.parse_args()
