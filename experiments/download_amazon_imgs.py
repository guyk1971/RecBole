# download amazon reviews images

import os
import pandas as pd
import json
from tqdm import tqdm
import gzip
import multiprocessing as mp
import wget
import subprocess
from functools import partial 

amz_dataset_root = '/home/gkoren2/datasets/recsys/amazon'

def load_data(meta_file_name):
    cached_filename = os.path.join(amz_dataset_root,os.path.basename(meta_file_name).split('.json')[0]+'.csv')
    if os.path.exists(cached_filename):
        df=pd.read_csv(cached_filename)
    else:
        data=[]
        with gzip.open(meta_file_name) as f:
            for l in tqdm(f):
                data.append(json.loads(l.strip()))
        df = pd.DataFrame.from_dict(data)
        df.to_csv(cached_filename,index=False)
    return df 


def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def get_img(save_dir,url):
    # wget.download(url,save_dir)
    print(url)
    cmd=f'wget -P {dirname} {url}'
    runcmd(cmd)

    return  






if __name__=='__main__':
    meta_file_name = 'meta_Clothing_Shoes_and_Jewelry.json.gz'
    print(f'downloading images from {meta_file_name}')

    meta_file =os.path.join(amz_dataset_root, 'meta_Clothing_Shoes_and_Jewelry.json.gz')

    print('loading the links data')
    df = load_data(meta_file)
    print('preparing the url list')
    url_list = [e for l in df.imageURLHighRes if isinstance(l,list) for e in l ]
    print(f'there are total of {len(url_list)} images to download')

    dirname = os.path.join(amz_dataset_root,os.path.splitext(os.path.basename(meta_file))[0])
    print(f'creating {dirname}')
    os.makedirs(dirname,exist_ok=True)

    get_img_2_dir = partial(get_img,dirname)

    n_processes= mp.cpu_count()
    pool = mp.Pool(n_processes)
    start_idx=1000001
    end_idx=1500000
    print(f'start downloading img index {start_idx} to {end_idx}')
    res = pool.map(get_img_2_dir,[url for url in url_list[start_idx:end_idx]])
