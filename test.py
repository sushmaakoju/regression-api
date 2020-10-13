import unittest
import os
import json
from reg import *

class RegTest(unittest.TestCase):
    def setUp(self):
        #self.app = api.app
        #self.client = self.app.test_client
        self.plotsobj = RegressionPlots()
        self.reg_obj =  Regression(self.plotsobj)
        #test default
        self.reg_obj.get_data_from_user("")
        self.reg_obj.clean_data()
        self.reg_obj.populate_fixtures('')
    
    def test_dtr(self):
        self.reg_obj.decisiontreeregr()
        self.assertIsNotNone(self.plotsobj.plots['dtr'])
        print(self.plotsobj.plots['dtr'])
        print(self.plotsobj.results['dtr_train'])
        print(self.plotsobj.results['dtr_test'])
    
    def test_svr(self):
        self.reg_obj.svr()
        self.assertIsNotNone(self.plotsobj.plots['svr'])
        print(self.plotsobj.plots['svr'])
        print(self.plotsobj.results['svr_train'])
        print(self.plotsobj.results['svr_test'])

    def test_rfr(self):
        self.reg_obj.rfr()
        self.assertIsNotNone(self.plotsobj.plots['rfr'])
        print(self.plotsobj.plots['rfr'])
        print(self.plotsobj.results['rfr_train'])

    def test_lasso(self):
        self.reg_obj.lr()
        self.assertIsNotNone(self.plotsobj.plots['lr'])
        print(self.plotsobj.plots['lr'])
        print(self.plotsobj.results['lr_test'])

    def test_br(self):
        self.reg_obj.br()
        self.assertIsNotNone(self.plotsobj.plots['br'])
        print(self.plotsobj.plots['br'])
        print(self.plotsobj.results['br_test'])
    
    def test_all(self):
        self.reg_obj.decisiontreeregr()
        self.reg_obj.svr()
        self.reg_obj.rfr()
        self.reg_obj.lr()
        self.reg_obj.br()
        self.reg_obj.compare_scores()
    
if __name__ == '__main__':
    unittest.main()

