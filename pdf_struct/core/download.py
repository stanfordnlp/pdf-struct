# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import contextlib
import hashlib
import os
import shutil
import sys
import tempfile

import filelock
import urllib

_dataset_root = os.environ.get(
    'PDFSTRUCT_DATASET_ROOT',
    os.path.join(os.path.expanduser('~'), '.pdf-struct', 'model'))


_url_root = os.environ.get(
    'PDFSTRUCT_URL_ROOT',
    'https://github.com/stanfordnlp/pdf-struct-models/raw/0.1.0/models/')


def get_cache_root():
    """Gets the path to the root directory to download and cache datasets.
    Returns:
        str: The path to the dataset root directory.
    """
    return _dataset_root


def get_model_url(model_name):
    return _url_root + model_name + '.joblib'


def get_cache_filename(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()


def cached_download(url):
    cache_root = get_cache_root()
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.isdir(cache_root):
            raise

    lock_path = os.path.join(cache_root, '_dl_lock')
    cache_path = os.path.join(cache_root, get_cache_filename(url))

    with filelock.FileLock(lock_path):
        if os.path.exists(cache_path):
            return cache_path

    with tempdir(dir=cache_root) as temp_root:
        temp_path = os.path.join(temp_root, 'dl')
        sys.stderr.write('Downloading from {}...\n'.format(url))
        sys.stderr.flush()
        urllib.request.urlretrieve(url, temp_path)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cache_path)

    return cache_path


def cached_model_download(model_name):
    url = get_model_url(model_name)
    try:
        path = cached_download(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        else:
            raise e
    return path



@contextlib.contextmanager
def tempdir(**kwargs):
    # A context manager that defines a lifetime of a temporary directory.
    ignore_errors = kwargs.pop('ignore_errors', False)

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=ignore_errors)
