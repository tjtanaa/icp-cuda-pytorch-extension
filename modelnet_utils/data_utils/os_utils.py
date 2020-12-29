import os
from os.path import join
import wget
import zipfile

def create_dir(path, clean=False):
    if os.path.isdir(path):
        if clean:
            os.system('rm -r {}'.format(path))
            os.mkdir(path)
            print("WARNING: dir {} has been cleaned.".format(path))
        else:
            print("WARNING: dir {} already exists, you want to clean it? [y/n]".format(path))
            answer = input()
            if answer == 'y':
                os.system('rm -r {}'.format(path))
                os.mkdir(path)
                print("WARNING: dir: {} has been cleaned.".format(path))
            elif answer == 'n':
                print("INFO: continue without cleaning.")
            else:
                raise ValueError("Process quit!")
    else:
        os.mkdir(path)


def download_zip(download_dir, url):
    file_name = url.split('/')[-1]
    file_dir = join(download_dir, file_name)
    if os.path.isfile(file_dir):
        print("INFO: using existing {} file: {}".format(file_name, file_dir))
    else:
        print("INFO: downloading {} from {}".format(file_name, url))
        wget.download(url, file_dir)
        print("\nINFO: {} successfully downloaded: {}".format(file_name, file_dir))


def unzip_file(download_dir, name):
    if not os.path.isdir(join(download_dir, name)):
        print("INFO: unzipping...")
        zip_ref = zipfile.ZipFile(join(download_dir, '{}.zip'.format(name)), 'r')
        zip_ref.extractall(download_dir)
        zip_ref.close()
    else:
        print("INFO: the file has already been unzipped.")