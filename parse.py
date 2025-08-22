import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MAGNeT")
    parser.add_argument('--batch_size', type=int,default=16,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--lr', type=float,default=0.0005,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--result_path', type=str,default="./result",
                        help="path to save results")
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--early_stop_patience', type=int, default=50, help='early stop patience')
    parser.add_argument('--use_env', action='store_true', help='whether use env data')
    parser.add_argument('--dataset_split', type=str, default='few_shot', help='method to split dataset') 
    parser.add_argument('--num_shot', type=int, default=1, help='how many few_shot data to use')
    parser.add_argument('--w_diverse', type=float, default=0.1, help='weight for diversity loss')
    parser.add_argument('--w_balance', type=float, default=0.01, help='weight for balance loss')
    # temperature
    parser.add_argument('--temperature', type=float, default=2., help='temperature for softmax')

    return parser.parse_args()