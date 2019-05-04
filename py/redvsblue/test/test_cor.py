import unittest
import os
import tempfile
import shutil
from pkg_resources import resource_filename
import sys
if (sys.version_info > (3, 0)):
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

        if self._test:
            self.remove_folder()

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

if __name__ == '__main__':
    unittest.main()
