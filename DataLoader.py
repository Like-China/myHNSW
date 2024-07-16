import numpy as np

class Loader:
    
    
    def __init__(self, dbname):
        self.dbname = dbname
    
    
    def read(self, dataPath):
        dbname = self.dbname
        if dbname == 'RANDOM':
            nb=100000
            d=64
            nq=100
            np.random.seed(1234)             # make reproducible
            xb = np.random.random((nb, d)).astype('float32')
            xb[:, 0] += np.arange(nb) / 1000.
            xq = np.random.random((nq, d)).astype('float32')
            xq[:, 0] += np.arange(nq) / 1000.
            xt=xb
        elif dbname=='SIFTSMALL':
            xt = fvecs_read(dataPath+"siftsmall/siftsmall_learn.fvecs")
            print('learn size',xt.shape)
            xb = fvecs_read(dataPath+"siftsmall/siftsmall_base.fvecs")
            print('base size',xb.shape)
            xq = fvecs_read(dataPath+"siftsmall/siftsmall_query.fvecs")
            print('query size',xq.shape)
            gt = ivecs_read(dataPath+"siftsmall/siftsmall_groundtruth.ivecs")
            print('groundtruth size',gt.shape)
        elif dbname == 'SIFT':
            xt = fvecs_read(dataPath+"sift1M/sift_learn.fvecs")
            print('learn size',xt.shape)
            xb = fvecs_read(dataPath+"sift1M/sift_base.fvecs")
            print('base size',xb.shape)
            xq = fvecs_read(dataPath+"sift1M/sift_query.fvecs")
            print('query size',xq.shape)
            gt = ivecs_read(dataPath+"sift1M/sift_groundtruth.ivecs")
            print('groundtruth size',gt.shape)
        elif dbname == 'GIST':
            xt = fvecs_read("../data/gist/gist_learn.fvecs")
            print('learn size',xt.shape)
            xb = fvecs_read("../data/gist/gist_base.fvecs")
            print('base size',xb.shape)
            xq = fvecs_read("../data/gist/gist_query.fvecs")
            print('query size',xq.shape)
            gt = ivecs_read("../data/gist/gist_groundtruth.ivecs")
            print('groundtruth size',gt.shape)
        elif dbname == 'Deep1M':
            xt = fvecs_read("../data/deep1M/deep1M_base.fvecs")
            print('learn size',xt.shape)
            xb = fvecs_read("../data/deep1M/deep1M_base.fvecs")
            print('base size',xb.shape)
            xq = fvecs_read("../data/deep1M/deep1B_queries.fvecs")
            print('query size',xq.shape)
            gt = ivecs_read("../data/deep1M/deep1M_groundtruth.ivecs")
            print('groundtruth size',gt.shape)
        elif dbname == 'Deep1B':
            #xt = fvecs_read("../data/deep1b/learn.fvecs")
            #xb = fvecs_read("../data/deep1b/base.fvecs")
            #xq = fvecs_read("../data/deep1b/deep1B_queries.fvecs")
            #gt = ivecs_read("../data/deep1b/deep1B_groundtruth.ivecs")
            
            xb = mmap_fvecs(dataPath+'deep1b/base.fvecs')
            
            xq = mmap_fvecs(dataPath+'deep1b/deep1B_queries.fvecs')
            xt = mmap_fvecs(dataPath+'deep1b/learn.fvecs')
            # deep1B's train is is outrageously big
            xt = xt[:1 * 1000 * 1000]
            #xb = xb[:1 * 1000 * 1000]
            #xt=xb
            gt = ivecs_read(dataPath+'deep1b/deep1B_groundtruth.ivecs')

            print('base size',xb.shape)
            print('learn size',xt.shape)
            print('query size',xq.shape)
            print('groundtruth size',gt.shape)
        elif dbname == 'Deep1B2':
            xb = fvecs_read("../data/deep1b/base.fvecs")
            xt = fvecs_read("../data/deep1b/learn.fvecs")
            xq = fvecs_read("../data/deep1b/deep1B_queries.fvecs")
            gt = ivecs_read("../data/deep1b/deep1B_groundtruth.ivecs")
            print('base size',xb.shape)
            print('learn size',xt.shape)
            print('query size',xq.shape)
            print('groundtruth size',gt.shape)
        elif dbname == 'bigann2':
            xb = fvecs_read(dataPath+'bigann/bigann_base.bvecs')
            xq = fvecs_read(dataPath+'bigann/bigann_query.bvecs')
            xt = fvecs_read(dataPath+'bigann/bigann_learn.bvecs')
            gt = ivecs_read(dataPath+'bigann/gnd/idx_1000M.ivecs')
            print('base size',xb.shape)
            print('learn size',xt.shape)
            print('query size',xq.shape)
            print('groundtruth size',gt.shape)
        elif 'bigann' in dbname :
            # SIFT1M to SIFT1000M
            dbsize =int(dbname[6:-1])
            print('dbsize:',dbsize)
            xb = mmap_bvecs(dataPath+'bigann/bigann_base.bvecs')
            xq = mmap_bvecs(dataPath+'bigann/bigann_query.bvecs')
            
            # trim xb to correct size
            xb = xb[:dbsize * 1000 * 1000]
            xt = mmap_bvecs(dataPath+'bigann/bigann_learn.bvecs')
            xt = xt[:1 * 1000 * 1000]
            gt = ivecs_read(dataPath+'bigann/gnd/idx_%dM.ivecs' % dbsize)

            print('base size',xb.shape)
            print('learn size',xt.shape)
            print('query size',xq.shape)
            print('groundtruth size',gt.shape)
        else:
            print('unknown dataset', dbname, file=sys.stderr)
            sys.exit(1)
        return xt, xb, xq, gt

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]
def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]
            
            