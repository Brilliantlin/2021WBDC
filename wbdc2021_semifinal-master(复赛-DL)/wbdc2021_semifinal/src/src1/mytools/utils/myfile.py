import os
import pickle
def removeFile(path):
    '''递归删除文件夹下所有'''
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
def makedirs(prefix):
    '''prefix：文件夹目录，可以递归生成'''
    if not os.path.exists(prefix):
        os.makedirs(prefix)

def load_stop_words(filename):
    with open(filename,'r',encoding='utf-8') as f:
        s = f.readlines()
    return s

def savePkl(config, filepath):
    f = open(filepath, 'wb')
    pickle.dump(config, f,protocol = 4)
    f.close()

def loadPkl(filepath):
    f = open(filepath, 'rb')
    config = pickle.load(f)
    return config