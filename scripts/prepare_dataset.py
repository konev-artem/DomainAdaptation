import os
from os.path import join, exists, splitext

import time
from shutil import move, rmtree
import platform


class DatasetPreparer:
    def __init__(self, dataset_name, dataset_root):
        print('Preparing {}...'.format(dataset_name.upper()))
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.dataset_dir = join(dataset_root, dataset_name)

    def download_dataset(self):
        pass

    def uncompress(self):
        pass

    def create_dataframe(self):
        pass

    def prepare_dataset(self):
        print('Step 1 / 2: downloading dataset (this may take a while)...')
        print('WARNING: all existing folders will be overwritten')
        for i in reversed(range(11)):
            print('New download starts in {:02d} s (press "^C" to exit)'.format(i), end='\r')
            time.sleep(1)
        print()
        self.download_dataset()

        print('Step 2 / 2: uncompressing dataset (this may take a while)...')
        self.uncompress()

        print('Completed')


class OfficeHomePreparer(DatasetPreparer):
    def __init__(self, dataset_name, dataset_root):
        super().__init__(dataset_name, dataset_root)

    def download_dataset(self):
        cookies_file = '/tmp/cookies.txt'

        sed = "gsed" if platform.system() == "Darwin" else "sed"

        confirm_cmd_tmp = "wget --quiet --save-cookies {cookies_file} --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B81rNlvomiwed0V1YUxQdC1uOTg' -O- | {sed} -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p'".format(
            sed=sed,
            cookies_file=cookies_file)
        confirm_code_tmp = "$({confirm_cmd})".format(confirm_cmd=confirm_cmd_tmp)
        url = "https://docs.google.com/uc?export=download&confirm={confirm_code}&id=0B81rNlvomiwed0V1YUxQdC1uOTg".format(
            confirm_code=confirm_code_tmp)
        download_cmd_tmp = 'wget --load-cookies {cookies_file} "{url}" -O {dataset_dir}.zip && rm -rf {cookies_file}'.format(
            url=url, dataset_dir=self.dataset_dir, cookies_file=cookies_file)
        os.system(download_cmd_tmp)

    def uncompress(self):
        import zipfile
        with zipfile.ZipFile(self.dataset_dir + '.zip', 'r') as zf:
            zip_name = zf.namelist()[0]
            zf.extractall(self.dataset_root)
        os.rename(join(self.dataset_root, zip_name), self.dataset_dir)
        os.remove(self.dataset_dir + '.zip')


class VisdaPreparer(DatasetPreparer):
    def __init__(self, dataset_name, dataset_root):
        super().__init__(dataset_name, dataset_root)

    def download_dataset(self):
        if os.path.exists(self.dataset_dir):
            rmtree(self.dataset_dir)
        os.makedirs(self.dataset_dir)

        cookies_file = '/tmp/cookies.txt'

        sed = "gsed" if platform.system() == "Darwin" else "sed"

        confirm_cmd_train = "wget --quiet --save-cookies {cookies_file} --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0BwcIeDbwQ0XmdENwQ3R4TUVTMHc' -O- | {sed} -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p'".format(
            sed=sed,
            cookies_file=cookies_file)
        confirm_code_train = "$({confirm_cmd})".format(confirm_cmd=confirm_cmd_train)
        url = "https://docs.google.com/uc?export=download&confirm={confirm_code}&id=0BwcIeDbwQ0XmdENwQ3R4TUVTMHc".format(
            confirm_code=confirm_code_train)
        download_cmd_train = 'wget --load-cookies {cookies_file} "{url}" -O {dataset_dir}/train.tar && rm -rf {cookies_file}'.format(
            url=url, dataset_dir=self.dataset_dir, cookies_file=cookies_file)
        print("--------------DOWNLOADING TRAIN DATASET--------------")
        os.system(download_cmd_train)

        confirm_cmd_valid = "wget --quiet --save-cookies {cookies_file} --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0BwcIeDbwQ0XmUEVJRjl4Tkd4bTA' -O- | {sed} -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p'".format(
            sed=sed,
            cookies_file=cookies_file)
        confirm_code_valid = "$({confirm_cmd})".format(confirm_cmd=confirm_cmd_valid)
        url = "https://docs.google.com/uc?export=download&confirm={confirm_code}&id=0BwcIeDbwQ0XmUEVJRjl4Tkd4bTA".format(
            confirm_code=confirm_code_valid)
        download_cmd_valid = 'wget --load-cookies {cookies_file} "{url}" -O {dataset_dir}/validation.tar && rm -rf {cookies_file}'.format(
            url=url, dataset_dir=self.dataset_dir, cookies_file=cookies_file)
        print("--------------DOWNLOADING VALIDATION DATASET--------------")

        os.system(download_cmd_valid)

    def uncompress(self):
        tar_cmd = "tar xvf {dataset_dir}/train.tar -C {dataset_dir}".format(dataset_dir=self.dataset_dir)
        os.system(tar_cmd)
        os.remove('{}/train.tar'.format(self.dataset_dir))

        tar_cmd = "tar xvf {dataset_dir}/validation.tar -C {dataset_dir}".format(dataset_dir=self.dataset_dir)
        os.system(tar_cmd)
        os.remove('{}/validation.tar'.format(self.dataset_dir))


class DomainNetPreparer(DatasetPreparer):
    def __init__(self, dataset_name, dataset_root):
        super().__init__(dataset_name, dataset_root)

    def download_dataset(self):
        # http://ai.bu.edu/M3SDA/
        if os.path.exists(self.dataset_dir):
            rmtree(self.dataset_dir)
        os.makedirs(self.dataset_dir)

        cookies_file = '/tmp/cookies.txt'

        download_cmd_clipart = "wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip -O {}/clipart.zip".format(
            self.dataset_dir)
        print("--------------DOWNLOADING Clipart DATASET--------------")
        os.system(download_cmd_clipart)

        download_cmd_infograph = "wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip -O {}/infograph.zip".format(
            self.dataset_dir)
        print("--------------DOWNLOADING Infograph DATASET--------------")
        os.system(download_cmd_infograph)

        download_cmd_painting = "wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip -O {}/painting.zip".format(
            self.dataset_dir)
        print("--------------DOWNLOADING Painting DATASET--------------")
        os.system(download_cmd_painting)

        download_cmd_quickdraw = "wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip -O {}/quickdraw.zip".format(
            self.dataset_dir)
        print("--------------DOWNLOADING Quickdraw DATASET--------------")
        os.system(download_cmd_quickdraw)

        download_cmd_real = "wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip -O {}/real.zip".format(
            self.dataset_dir)
        print("--------------DOWNLOADING Real DATASET--------------")
        os.system(download_cmd_real)

        download_cmd_sketch = "wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip -O {}/sketch.zip".format(
            self.dataset_dir)
        print("--------------DOWNLOADING Sketch DATASET--------------")
        os.system(download_cmd_sketch)

    def uncompress(self):
        import zipfile

        import glob, os
        os.chdir(self.dataset_dir + '/')
        for file in glob.glob("*.zip"):
            with zipfile.ZipFile(file, 'r') as zf:
                zf.extractall(self.dataset_dir)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('dataset', type=str,
                        choices=['office_home', 'office_31', 'visda_2017', 'domain_net', 'digits'],
                        help='Dataset name')
    parser.add_argument('--dataset_root', type=str, default=os.getcwd(),
                        help='Path to dataset root (default: ".")')
    # parser.add_argument('')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'office_home':
        Preparer = OfficeHomePreparer
    elif args.dataset == 'office_31':
        raise NotImplemented
    elif args.dataset == 'visda_2017':
        Preparer = VisdaPreparer
    elif args.dataset == 'domain_net':
        Preparer = DomainNetPreparer
    elif args.dataset == "digits":
        # https://domainadaptation.org/api/salad.datasets.digits.html
        raise NotImplemented
    else:
        raise NotImplemented

    dataset_preparer = Preparer(args.dataset, args.dataset_root)

    dataset_preparer.prepare_dataset()
