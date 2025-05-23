import time
import argparse
import logging

def str_to_bool(value):
    # Convert string input to a boolean value
    if value.lower() in {'true', '1', 'yes', 'y'}:
        return True
    elif value.lower() in {'false', '0', 'no', 'n'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: '{value}'")

parser = argparse.ArgumentParser()
parser.add_argument('--running_time', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sample_interval', type=int, default=1)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, nargs='+', default=[1e-4, 1e-4, 1e-3, 1e-4, 2e-4]) # xmedianet [1e-4, 1e-4, 3e-4, 1e-4, 1e-4] xmedia [1e-4, 1e-4, 1e-3, 1e-4, 2e-4]
parser.add_argument('--batch_size', type=int, default=256)     
parser.add_argument('--bits', type=int, default=128)   
parser.add_argument('--alpha', type=float, default=1) # dhl
parser.add_argument('--beta1', type=float, default=1) # klr
parser.add_argument('--beta2', type=float, default=1) # klt
parser.add_argument('--eta', type=float, default=1) # klm
parser.add_argument('--bound', type=float, default=0.95) # similarity bound
parser.add_argument('--q', type=float, default=0.01) # 0.01
parser.add_argument('--num_prototypes_per_class', type=int, default=3) #
parser.add_argument("--tau", type=float, default=1.0) # temperature
parser.add_argument('--datasets', type=str, default='xmedia')  # wiki_old xmedia, xmedianet, nus
parser.add_argument('--view_id', type=int, default=-1)
parser.add_argument('--epochs', type=int, default=300) # 200 300
parser.add_argument('--just_val', type=bool, default=False)
parser.add_argument('--logging', type=str, default=None)  # wiki_old xmedia, xmedianet, nus
parser.add_argument('--first_modality', type=int, default=0)
parser.add_argument('--is_fine_tuning', type=str_to_bool, default=True)
print("current local time: ", time.asctime(time.localtime(time.time())))
import os
args = parser.parse_args()

seed = 1000
from to_seed import to_seed
to_seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)
# 配置日志记录器
logger_name = args.logging if args.logging else args.datasets
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('logging/' + logger_name + '.log'),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

def main():
    logger.info('Bit: ' + str(args.bits))
    logger.info(args)
    from SHE import Solver
    solver = Solver(args, logger)
    # solver.train_score()
    solver.train()
    exit()

if __name__ == '__main__':
    main()