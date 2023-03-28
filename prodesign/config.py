import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu_list', type=int, nargs='+',
      help='list of GPU IDs')
parser.add_argument('--device', type=str, default='cuda',
      help='set device')

parser.add_argument('--train_data_dir', type=str,default='examples/train',
      help='train data directory')
parser.add_argument('--test_data_dir', type=str,default='examples/test',
      help='test data directory')
parser.add_argument('-b', '--batch_size', type=int, default=1000,
      help='batch size, default=1000')
#parser.add_argument('--num_workers', type=int, default=1,
#      help='number of workers, default=1')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
      help='learning rate, default=1e-3')
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-d', '--dim', type=int, default=256)
parser.add_argument('--prefix', type=str, default='log')
parser.add_argument('--checkpoint', type=str, default=None,
      help='path to save or load checkpoint')
parser.add_argument('--save_file', type=str, default=None,
      help='path to save or load model')
parser.add_argument('--pdb_file', type=str, default=None,
      help='pdb_file to design')

args = parser.parse_args()
