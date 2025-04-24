import os
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F

import sqlite3
import io
import json
from datetime import datetime
import re
import pickle

from feature_extractor import feature_extractor, valid_exts

def hexstr_to_bitarray(hexstr):
    scale = 16
    num_of_bits = len(hexstr) * 4
    bin_str = bin(int(hexstr, scale))[2:].zfill(num_of_bits)
    return np.array(list(map(int, bin_str)))

# numpy array를 BLOB으로 변환
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

# BLOB을 numpy array로 변환
def convert_array(blob):
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


conn = sqlite3.connect('./filelist.db')
cursor = conn.cursor()

# filelist 테이블 생성 쿼리
create_table_query = '''
CREATE TABLE IF NOT EXISTS filelist (
    filepath TEXT UNIQUE,
    latent array,
    ahash TEXT,
    dhash TEXT,
    dt TEXT,
    fsize LONG
);
'''

cursor.execute(create_table_query)
conn.commit()
conn.close()


d_con = sqlite3.connect('./filelist.db')
s_cons = [sqlite3.connect(fn) for fn in ['./filelist1.db', './filelist2.db', './filelist3.db']]

d_cur = d_con.cursor()
s_curs = [s_con.cursor() for s_con in s_cons]

for s_cur in s_curs:
    s_cur.execute("SELECT * FROM filelist")
    for s_row in tqdm(s_cur.fetchall()):
        fn = s_row[0]
        latent = s_row[1]
        ahash = s_row[2]
        dhash = s_row[3]
        dt = s_row[4]
        fsize = s_row[5]
        if dt[:10] >= '2025:04:23':
            print(f'{fn}: read={dt}')
            dt = ''
        elif fn.startswith('/mnt/d/photo/'):
            matches = re.findall(r'\d{4}/', fn)
            if len(matches) > 0:
                s = matches[0].replace('/', '')
                if dt[:4] != s:
                    print(f'{fn}: read={dt}, desired={s}')
                    dt = ''
        d_cur.execute(
            'INSERT INTO filelist (filepath, latent, ahash, dhash, dt, fsize) VALUES (?, ?, ?, ?, ?, ?)',
            (fn, latent, ahash, dhash, dt, fsize)
        )
        d_con.commit()

for s_con in s_cons:
    s_con.close()
d_con.close()



con = sqlite3.connect('./filelist.db')
cur = con.cursor()

cur.execute("SELECT * FROM filelist WHERE filepath LIKE '/mnt/d/photo/Takeout/%'")
s_rows = cur.fetchall()
cur.execute("SELECT * FROM filelist WHERE filepath NOT LIKE '/mnt/d/photo/Takeout/%'")
t_rows = cur.fetchall()

len(s_rows), len(t_rows)



tra = []
trd = []
for tr in t_rows:
    tra.append(hexstr_to_bitarray(tr[2]))
    trd.append(hexstr_to_bitarray(tr[3]))
tra = np.vstack(tra)
trd = np.vstack(trd)

tra.shape, trd.shape



del_dict = {}
N = tra.shape[0]

for sr in tqdm(s_rows):
    sra = hexstr_to_bitarray(sr[2])
    sra = np.tile(sra, (N, 1))
    sima = np.sum(sra != tra, axis=1)
    mask = sima < 4
    srd = hexstr_to_bitarray(sr[3])
    srd = np.tile(srd, (N, 1))
    simd = np.sum(srd != trd, axis=1)
    mask &= simd < 4
    if mask.any():
        print(f'{sr[0]} <{sr[4]}, {sr[5]}>')
        for idx in np.where(mask)[0]:
            if sr[0] != t_rows[idx][0]:
                print(f'\t{t_rows[idx][0]} <{t_rows[idx][4]}, {t_rows[idx][5]}> : a({sima[idx]}), d({simd[idx]})')
        del_dict[sr[0]] = [(t_rows[idx][0], sima[idx], simd[idx]) for idx in np.where(mask)[0] if t_rows[idx][0] != sr[0]]




with open('del_dict.pkl', 'wb') as f:
    pickle.dump(del_dict, f)
len(del_dict)



with open('del_dict.pkl', 'rb') as f:
    del_dict = pickle.load(f)
len(del_dict)



conn = sqlite3.connect('./filelist.db')
cursor = conn.cursor()
cursor.execute("SELECT filepath, dt, fsize FROM filelist")
rows = cursor.fetchall()
file_info_dict = {}
for row in rows:
    file_info_dict[row[0]] = (row[1], row[2])
conn.close()



del_files = []
undel_files = []
for fn, childs in tqdm(del_dict.items()):
    remove = False
    pn = '/'.join(fn.split('/')[:-1])
    dt, fsize = file_info_dict[fn]
    ext = fn.split('.')[-1].lower()
    for child in childs:
        cfn = child[0]
        cpn = '/'.join(cfn.split('/')[:-1])
        cdt, cfsize = file_info_dict[cfn]
        cext = cfn.split('.')[-1].lower()
        if len(cdt) > 0:
            remove = True
            break
    if remove:
        print(f'delete: {fn}:{dt} <- {cfn}:{cdt}')
        del_files.append(fn)
    else:
        undel_files.append(fn)

len(del_files), len(undel_files)



con = sqlite3.connect('./filelist.db')
cur = con.cursor()
for fn in tqdm(del_files):
    if os.path.exists(fn):
        os.remove(fn)
    cur.execute("DELETE FROM filelist WHERE filepath = ?", (fn,))
    con.commit()
con.close()

new_del_dict = {}
for fn in undel_files:
    new_del_dict[fn] = del_dict[fn]
del_dict = new_del_dict

len(del_dict)



