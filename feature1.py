import os
import numpy as np
from glob import glob
from tqdm.notebook import tqdm

import sqlite3
import io

from feature_extractor import feature_extractor, valid_exts



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



conn = sqlite3.connect('./filelist1.db')
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



conn = sqlite3.connect('./filelist1.db')

def filepath_exists(conn, filepath):
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM filelist WHERE filepath = ?', (filepath,))
    exists = cursor.fetchone() is not None
    return exists

def insert_file(conn, filepath, latent, ahash, dhash, dt, filesize):
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO filelist (filepath, latent, ahash, dhash, dt, fsize) VALUES (?, ?, ?, ?, ?, ?)',
        (filepath, latent, ahash, dhash, dt, filesize)
    )
    conn.commit()

def has_datetime(conn, filepath):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM filelist WHERE filepath = ? AND dt != '' LIMIT 1",
        (filepath,)
    )
    result = cursor.fetchone()
    return result is not None

def update_datetime(conn, filepath, dt):
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE filelist SET dt = ? WHERE filepath = ?",
        (dt, filepath)
    )
    conn.commit()

FE = feature_extractor()



for fn in tqdm(glob('/mnt/d/album/**/*.*', recursive=True)):
    if os.path.isfile(fn):
        ext = fn.split('.')[-1].lower()
        if ext in valid_exts:
            if not filepath_exists(conn, fn):
                latent, ahash, dhash, dt, filesize = FE.to_features(fn)
                latent = latent.detach().numpy()
                ahash = str(ahash)
                dhash = str(dhash)
                insert_file(conn, fn, latent, ahash, dhash, dt, filesize)


