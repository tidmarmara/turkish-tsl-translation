# import _init_paths
from lib.utils import Dataset
from lib.opts import Opts

def main(opts):
    dataLoader = Dataset(opts)
    datasets = dataLoader.create_dataset()
    

if __name__ == '__main__':
    opts = Opts().parse()
    main(opts)


