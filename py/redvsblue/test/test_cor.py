import unittest
import os
import tempfile
import shutil
from pkg_resources import resource_filename
import subprocess
import fitsio
import scipy as sp

class TestCor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._branchFiles = tempfile.mkdtemp()+"/"

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)

    def test_cor(self):

        self._test = True
        self.redvsblue_base = resource_filename('redvsblue', './').replace('py/redvsblue/./','')
        self.send_requirements()
        self._masterFiles = self.redvsblue_base+'/py/redvsblue/test/data/'
        self.produce_folder()
        self.send_linefitter_SDSS()
        self.send_linefitter_DESI()

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
        dic = {'THING_ID':sp.array([113373895,242548124]),'PLATE':sp.array([4300,5138]),
            'MJD':sp.array([55528,55830]),'FIBERID':sp.array([224,20]),'Z':sp.array([1.24624,2.87229])}
        out = fitsio.FITS(self._branchFiles+'/Products/cat.fits','rw',clobber=True)
        out.write( [v for v in dic.values()] ,names=[k for k in dic.keys()],extname='CAT')
        out.close()

        return
    def remove_folder(self):
        """
            Remove the produced folders
        """

        print("\n")
        shutil.rmtree(self._branchFiles, ignore_errors=True)

        return
    def compare_fits(self,path1,path2,nameRun=""):

        print("\n")
        m = fitsio.FITS(path1)
        self.assertTrue(os.path.isfile(path2),"{}".format(nameRun))
        b = fitsio.FITS(path2)

        self.assertEqual(len(m),len(b),"{}".format(nameRun))

        for i,_ in enumerate(m):
            if i==0:
                continue

            ###
            ld_m = sorted(m[i].get_colnames())
            ld_b = sorted(b[i].get_colnames())
            self.assertListEqual(ld_m,ld_b,"{}".format(nameRun))

            for k in ld_m:
                d_m = m[i][k][:]
                d_b = b[i][k][:]
                if d_m.dtype in ['<U23','S23']: ### for fitsio old version compatibility
                    d_m = sp.char.strip(d_m)
                if d_b.dtype in ['<U23','S23']: ### for fitsio old version compatibility
                    d_b = sp.char.strip(d_b)
                self.assertEqual(d_m.size,d_b.size,"{}: Header key is {}".format(nameRun,k))
                if not sp.array_equal(d_m,d_b):
                    print("WARNING: {}: {}, Header key is {}, arrays are not exactly equal, using allclose".format(nameRun,i,k))
                    diff = d_m-d_b
                    w = d_m!=0.
                    diff[w] = sp.absolute( diff[w]/d_m[w] )
                    allclose = sp.allclose(d_m,d_b)
                    self.assertTrue(allclose,"{}: Header key is {}, maximum relative difference is {}".format(nameRun,k,diff.max()))

        m.close()
        b.close()

        return
    def load_requirements(self):

        req = {}

        path = self.redvsblue_base+'/requirements.txt'
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
    def send_linefitter_SDSS(self):

        ###
        print("\n")
        cmd = 'redvsblue_lineFitter.py --out '+self._branchFiles+'/Products/lineFitter.fits'
        cmd += ' --in-dir '+self._masterFiles+'/data/sdss/'
        cmd += ' --drq '+self._branchFiles+'/Products/cat.fits'
        cmd += ' --z-key Z'
        cmd += ' --qso-pca '+self.redvsblue_base+'/etc/rrtemplate-qso.fits'
        cmd += ' --stack-obs'
        cmd += ' --nproc 1'
        cmd += ' --no-extinction-correction'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + '/Products/lineFitter.fits'
            path2 = self._branchFiles + '/Products/lineFitter.fits'
            self.compare_fits(path1,path2,"redvsblue_lineFitter_SDSS.py")

        return
    def send_linefitter_DESI(self):

        ###
        print("\n")
        cmd = 'redvsblue_lineFitter.py --out '+self._branchFiles+'/Products/lineFitter_DESI.fits'
        cmd += ' --in-dir '+self._masterFiles+'/data/desi/spectra-16/'
        cmd += ' --drq '+self._masterFiles+'/data/desi/zcat.fits'
        cmd += ' --z-key Z'
        cmd += ' --qso-pca '+self.redvsblue_base+'/etc/rrtemplate-qso.fits'
        cmd += ' --nproc 1'
        cmd += ' --no-extinction-correction'
        cmd += ' --data-format DESI'
        subprocess.call(cmd, shell=True)

        ### Test
        if self._test:
            path1 = self._masterFiles + '/Products/lineFitter_DESI.fits'
            path2 = self._branchFiles + '/Products/lineFitter_DESI.fits'
            self.compare_fits(path1,path2,"redvsblue_lineFitter_DESI.py")

        return

if __name__ == '__main__':
    unittest.main()
