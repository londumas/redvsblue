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

        self._test = True
        self.send_requirements()
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
        path_to_data = 'https://dr14.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/'
        dic = {'THING_ID':sp.array([113373895,242548124]),'PLATE':sp.array([4300,5138]),
            'MJD':sp.array([55528,55830]),'FIBERID':sp.array([224,20]),'Z':sp.array([1.24624,2.87229])}

        for i in range(len(dic['PLATE'])):
            p = dic['PLATE'][i]
            m = dic['MJD'][i]
            f = dic['FIBERID'][i]
            path = '{}/{}/spec-{}-{}-{}.fits'.format(path_to_data,str(p).zfill(4),str(p).zfill(4),m,str(f).zfill(4))
            path2 = self._branchFiles+'/data/spectra/lite/{}/'.format(str(p).zfill(4))
            path3 = path2+'/spec-{}-{}-{}.fits'.format(str(p).zfill(4),m,str(f).zfill(4))
            print(path)
            if not os.path.isfile(path3):
                wget.download(url=path,out=path2)

        ### Save
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

            ###
            r_m = m[i].read_header().records()
            ld_m = []
            for el in r_m:
                name = el['name']
                if len(name)>5 and name[:5]=="TTYPE":
                    ld_m += [el['value'].replace(" ","")]
            ###
            r_b = b[i].read_header().records()
            ld_b = []
            for el in r_b:
                name = el['name']
                if len(name)>5 and name[:5]=="TTYPE":
                    ld_b += [el['value'].replace(" ","")]

            print(ld_m)
            print(ld_b)
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
                    print("WARNING: {}: Header key is {}, arrays are not exactly equal, using allclose".format(nameRun,k))
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

        ###
        cmd = 'redvsblue_lineFitter.py --out '+self._branchFiles+'/Products/lineFitter.fits'
        cmd += ' --in-dir '+self._branchFiles+'/data/'
        cmd += ' --drq '+self._branchFiles+'/Products/cat.fits'
        cmd += ' --z-key Z'
        cmd += ' --qso-pca '+resource_filename('redvsblue', '/../../etc/rrtemplate-qso.fits')
        cmd += ' --stack-obs'
        cmd += ' --nproc 1'
        cmd += ' --no-extinction-correction'
        subprocess.call(cmd, shell=True)


        ### Test
        if self._test:
            path1 = self._masterFiles + '/Products/lineFitter.fits'
            path2 = self._branchFiles + '/Products/lineFitter.fits'
            self.compare_fits(path1,path2,"redvsblue_lineFitter.py")

        return

if __name__ == '__main__':
    unittest.main()
