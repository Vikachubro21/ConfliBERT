import json
import pandas as pd
pd.set_option("max_colwidth", 600)
import ast
from bs4 import BeautifulSoup
import re
import requests
import time
import numpy as np
import zipfile
import os
import html
import re
import glob
import pathlib
import unicodedata
import tarfile

from pandarallel import pandarallel
pandarallel.initialize()
from unidecode import unidecode

def get_csv_size(csv_name):
    get_size = os.path.getsize(os.getcwd() + '/'+csv_name)
    mb_size = get_size/(1024 * 1024)
    mb_size = round(mb_size,1)
    return mb_size

def get_attribute(filename):
    if filename.endswith('csv'):
        df1 = pd.read_csv(filename,header=[0])
    else:
        df1=pd.read_json(filename,orient="records", lines=True)
    return list(df1.columns) 

def split_large_file(filename, source, output,size=None):
    df1=pd.read_csv(filename,header=[0])
    if size ==None:
        size = get_csv_size(filename)
    num_chunks = size//20
    if num_chunks == 0:
        num_chunks = 1 
    df_all = np.array_split(df1, num_chunks)

    for idx, file in enumerate(df_all):
        file.to_csv('%s/%s_%03d.csv'%(output, source, idx), index=False)   
        
def show_all_files(folder):
    df = pd.DataFrame(glob.glob('%s/*'%folder), columns = ['path'])
    df['root'] = df.path.apply(lambda x: x.split('/')[0])
    df['source'] = df.path.apply(lambda x: x.replace('%s/'%folder,'').split('_')[0])
    df['filename'] = df.path.apply(lambda x: x.replace('%s/'%folder,'').split('/')[-1])
    df = df.sort_values('source').reset_index(drop=True)
    df['size'] = df['path'].parallel_apply(get_csv_size)
    return df

def unicodetoascii(text):
    TEXT = (text.
    		replace('\\xe2\\x80\\x99', "'").
            replace('\\xc3\\xa9', 'e').
            replace('\\xe2\\x80\\x90', '-').
            replace('\\xe2\\x80\\x91', '-').
            replace('\\xe2\\x80\\x92', '-').
            replace('\\xe2\\x80\\x93', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x98', "'").
            replace('\\xe2\\x80\\x9b', "'").
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9d', '"').
            replace('\\xe2\\x80\\x9e', '"').
            replace('\\xe2\\x80\\x9f', '"').
            replace('\\xe2\\x80\\xa6', '...').#
            replace('\\xe2\\x80\\xb2', "'").
            replace('\\xe2\\x80\\xb3', "'").
            replace('\\xe2\\x80\\xb4', "'").
            replace('\\xe2\\x80\\xb5', "'").
            replace('\\xe2\\x80\\xb6', "'").
            replace('\\xe2\\x80\\xb7', "'").
            replace('\\xe2\\x81\\xba', "+").
            replace('\\xe2\\x81\\xbb', "-").
            replace('\\xe2\\x81\\xbc', "=").
            replace('\\xe2\\x81\\xbd', "(").
            replace('\\xe2\\x81\\xbe', ")"))
    return TEXT
pandarallel.initialize(nb_workers=8, progress_bar=False) 

df = pd.DataFrame(glob.glob('El-Panama-America_final/*/*.csv'), columns = ['path'])
df['source'] = df.path.apply(lambda x: x.split('/')[1].replace('.csv',''))
df = df.sort_values('source').reset_index(drop=True)
df['attributes']= df.path.parallel_apply(get_attribute)
df['size']= df.path.parallel_apply(get_csv_size)
pandarallel.initialize(nb_workers=8, progress_bar=False)       
_ = df.parallel_apply(lambda x: split_large_file(x['path'],x['source'],'split'), axis=1) 
df = show_all_files('split')
df = show_all_files('split')
# df['attribute'] = df.path.parallel_apply(get_attribute)
df.source.unique()
print('total size:', df['size'].sum())
df.groupby(by=["source"], dropna=False).sum()
def basic_process(filename, output_folder):
    
    sizes = []
    
    df1 = pd.read_csv(filename, header=[0])
    sizes.append(df1.shape[0])
    
    if 'url' not in df1.columns:
        df1['url'] ='\n'
    if 'title' not in df1.columns:
        df1['title']='\n'
    
#     df1 = df1[~df1.url.isnull()]
    
    df1 = df1[~df1.text.isnull()]
    df1 = df1[~df1.text.duplicated()]
#     df1 = df1[~df1.url.duplicated()]
    
    df1.loc[df1.title.isnull(),'title']='\n'
    df1.loc[df1.url.isnull(),'url']='\n'
    if 'abstract' in df1:
        df1.loc[df1.abstract.isnull(),'abstract']='\n'
        
    # ------------------- Start cleaning  --------------------------#
    
    # Convert coding
    df1.text = df1.text.apply(lambda x: unicodetoascii(x))
    df1.text = df1.text.apply(lambda x: unicodedata.normalize("NFKD", x))
    
    # email
    df1.text = df1.text.apply(lambda x: re.sub("\S+@\S+(?:\.\S+)+",'',x))
    
    # telphone
    df1.text = df1.text.apply(lambda x: re.sub('\(\+( |-|\d)+\)( |-|\d)+',' ',x))
    df1.text = df1.text.apply(lambda x: re.sub('\+( |-|\d)+',' ',x))
    
    # noise
    df1.text =\
    df1.text.apply(lambda x: re.sub('\n(ad|advertisement|tweet):?\n', "", x, flags=re.IGNORECASE))
    
    # urls
    df1.text = df1.text.apply(lambda x: re.sub(r"http\S+", "", x))
    
    # delete too many \n
    df1.text = df1.text.apply(lambda x: re.sub('\n\n+', "\n\n", x, flags=re.IGNORECASE))
    
    # head and tails
    df1.text = df1.text.apply(lambda x: re.sub("^\s+|\s+$", "", x, flags=re.UNICODE)) 
    
    
    df1 = df1[df1.text.str.len()>100]
    
    sizes.append(df1.shape[0])
    
    # ------------------- Ending cleaning  --------------------------#
    
    filename = filename.split('/')[1]
    new_filename = output_folder + '/'+ filename

    print('%s:\t%s'%(filename, sizes))
    df1.to_csv(new_filename, index= False)
    return
pandarallel.initialize(nb_workers=8, progress_bar=False) 
_ = df['path'].parallel_apply(basic_process, output_folder='step1')
df = show_all_files('step1')
print('total size:', df['size'].sum())
df.groupby(by=["source"], dropna=False).sum()
# folder = '2.Organization'
# df = pd.DataFrame(columns=['path','source', 'filename'])
# df.path = [str(x) for x in pathlib.Path('%s/*'%folder).glob('**/*')]
df = pd.DataFrame(glob.glob('*/*/*'), columns = ['path'])
# df['root'] = df.path.apply(lambda x: x.split('/')[0])

df['source'] = df.path.apply(lambda x: x.split('/')[1])
df['filename'] = df.path.apply(lambda x: x.split('/')[-1])

# df = df.sort_values('source').reset_index(drop=True)
# df['size'] = df['path'].parallel_apply(get_csv_size)
# df['filename'] = df.path.apply(lambda x: x.replace('%s/'%folder,''))

df['json_file'] = df['filename'].apply(lambda x: x.replace('.csv', '.json'))

df['json_file'] = df.source +'/'+ df.json_file
df['tar_file'] = 'tar/'+ df.json_file+'.tar.gz'
df.json_file = 'json/'+ df.json_file
cwd = os.getcwd()
for i in df.source.unique():
    os.mkdir(cwd + '/json/'+i)
    os.mkdir(cwd + '/tar/'+i)  
df1 = pd.read_csv(df.path[0], header=[0])
def convert_json_tar(filename, json_file, tar_file):
    df1 = pd.read_csv(filename, header=[0])      
    df1.to_json(json_file, orient="records", lines=True)
    
    with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(json_file, arcname=os.path.basename(json_file))

pandarallel.initialize(nb_workers=8, progress_bar=True) 
_ = df.parallel_apply(lambda x: convert_json_tar(x['path'], x['json_file'], x['tar_file']), axis=1)
pandarallel.initialize() 

df.source = df.path.apply(lambda x: x.split('/')[1]).values
df['size'] = df['path'].parallel_apply(get_csv_size)
print(df['size'].sum())
df.groupby(by=["source"], dropna=False)['size'].sum().round(1)
