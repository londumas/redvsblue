import unittest
import os
import tempfile
import shutil
from pkg_resources import resource_filename
import sys
import wget
import subprocess
import fitsio
import scipy as sp
if sys.version_info>(3,0):
    # Python 3 code in this block
    import configparser as ConfigParser
else:
    import ConfigParser

class TestCor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._branchFiles = tempfile.mkdtemp()+"/"

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)

    def test_cor(self):

        self._test = False
        self.send_requirements()
        self._branchFiles = '/uufs/astro.utah.edu/common/home/u6011908/Programs/londumas/redvsblue/py/redvsblue/test/data_test/'
        self._masterFiles = resource_filename('redvsblue', 'test/data/')
        self.produce_folder()
        self.send_linefitter()

        if self._test:
            self.remove_folder()

        return
    def produce_folder(self):
        """
            Create the necessary folders
        """

        print("\n")
        lst_fold = ['/data/','/data/spectra/','/data/spectra/lite/',
            '/data/spectra/lite/4300/','/data/spectra/lite/5138/',
            '/Products/']

        for fold in lst_fold:
            if not os.path.isdir(self._branchFiles+fold):
                os.mkdir(self._branchFiles+fold)

        ###
        urls = {'4300':'https://dr14.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/4300/spec-4300-55528-0224.fits',
            '5138':'https://dr14.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/5138/spec-5138-55830-0020.fits'}
        for n,p in urls.items():
            print(self._branchFiles+'/data/spectra/lite/{}/{}'.format(n,p.split('/')[-1]))
            if not os.path.isfile(self._branchFiles+'/data/spectra/lite/{}/{}'.format(n,p.split('/')[-1])):
                wget.download(url=p,out=self._branchFiles+'/data/spectra/lite/{}/'.format(n))

        ###
        plate = sp.array([4300,5138])
        mjd = sp.array([55528,55830])
        fid = sp.array([224,20])
        thid = sp.array([0,1])
        zqso = sp.array([1.24624,2.87229])

        ### Save
        out = fitsio.FITS(self._branchFiles+'/Products/cat.fits','rw',clobber=True)
        cols=[thid,plate,mjd,fid,zqso]
        names=['THING_ID','PLATE','MJD','FIBERID','Z']
        out.write(cols,names=names,extname='CAT')
        out.close()

        return
    def remove_folder(self):
        """
            Remove the produced folders
        """

        print("\n")
        shutil.rmtree(self._branchFiles, ignore_errors=True)

        return
    def load_requirements(self):

        req = {}

        path = resource_filename('redvsblue', '/../../requirements.txt')
        with open(path,'r') as f:
            for l in f:
                l = l.replace('\n','').replace('==',' ').replace('>=',' ').split()
                self.assertTrue(len(l)==2,"requirements.txt attribute is not valid: {}".format(str(l)))
                req[l[0]] = l[1]

        return req



    def send_requirements(self):

        print("\n")
        req = self.load_requirements()
        for req_lib, req_ver in req.items():
            try:
                local_ver = __import__(req_lib).__version__
                if local_ver!=req_ver:
                    print("WARNING: The local version of {}: {} is different from the required version: {}".format(req_lib,local_ver,req_ver))
            except ImportError:
                print("WARNING: Module {} can't be found".format(req_lib))

        return
    def send_linefitter(self):

        cmd = 'redvsblue_lineFitter.py --out '+self._branchFiles+'/Products/lineFitter.fits'
        cmd += ' --in-dir '+self._branchFiles+'/data/'
        cmd += ' --drq '+self._branchFiles+'/Products/cat.fits'
        cmd += ' --z-key Z'
        cmd += ' --qso-pca '+resource_filename('redvsblue', '/../../etc/rrtemplate-qso.fits')
        cmd += ' --stack-obs'
        cmd += ' --no-extinction-correction'
        subprocess.call(cmd, shell=True)

        return

if __name__ == '__main__':
    unittest.main()
