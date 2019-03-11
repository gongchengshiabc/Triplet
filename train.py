import os,torch
from network import SBIR
from data import  DatasetProcessing,get_data_loader
import torchvision.transforms as transforms
from evaluate import CalcReMap,CalcReAcc
from optparse import OptionParser
def main(argv=None):
    #parser=argparse.ArgumentParser()
    parser = OptionParser(usage="""usage: %prog [options]""")
    parser.add_option('--rootpath',default='/home/zlm/VisualSearch/',type=str,help='Rootpath')
    parser.add_option('--collection',default='Sketchy',type=str,help='list of test list')
    parser.add_option('--model_dir',default='models',type=str,help='model dir')
    parser.add_option('--embed_size', default=256, type=int, help='embed_size')
    parser.add_option('--isnorm', default=0, type=int, help='l2norm')
    parser.add_option('--margin', default=15, type=float, help='margin')
    parser.add_option('--measure', default='cosine', type=str, help='measure ways')
    parser.add_option('--cost_style', default='mean', type=str, help='mean or sum')
    parser.add_option('--optimizer', default='SGD', type=str, help='SGD,Rmspror or Adam')
    parser.add_option('--loss', default='triplet', type=str, help='loss funcation')
    parser.add_option('--learning_rate', default=0.00001, type=float, help='learning')
    parser.add_option('--momentum', default=0.9, type=float, help='momentum')
    parser.add_option('--weight_decay', default=0.02, type=float, help='momentum')
    parser.add_option('--weight', default=100, type=float, help='the weight of softmax loss')
    parser.add_option('--gamma', default=0.1, type=float, help='gamma')
    parser.add_option('--step_size', default=10, type=int, help='step_size')
    parser.add_option('--num_epoch', default=10, type=int, help='the num of epoch')
    parser.add_option('--batch_size', default=8, type=int, help='the number of batch size')
    parser.add_option('--max_violation', default=1, type=int, help='useing max loss to train')
    opt = parser.parse_args()

    DATA_DIR = os.path.join(opt.rootpath, opt.collection, 'fine-grained', 'all')
    TRAIN_FILE = os.path.join(opt.rootpath, opt.collection, 'fine-grained', 'train.txt')
    TEST_FILE = os.path.join(opt.rootpath, opt.collection, 'fine-grained', 'test.txt')
    nclasses = 125
    TRAIN_LABEL = os.path.join(opt.rootpath, opt.collection, 'fine-grained', 'train_label.txt')
    TEST_LABEL = os.path.join(opt.rootpath, opt.collection, 'fine-grained', 'test_label.txt')
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
