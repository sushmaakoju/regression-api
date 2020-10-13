import unittest
import os
import json
from app import *

class RegApiTest(unittest.TestCase):

    def setUp(self):
        self.app = api.app
        self.client = self.app.test_client
    
    def test_start(self):
        result = self.client().post('/')
        self.assertEqual(result.status_code, 200)
        print(result.data)
        self.assertEqual(result.json['result'], 'Welcome to Regression api!')
    
    def test_train_svr(self):
        result = self.client().post('/train_svr')
        self.assertEqual(result.status_code, 200)

    def test_train_rfr(self):
        result = self.client().post('/train_rfr')
        self.assertEqual(result.status_code, 200)
    
    def test_train_lr(self):
        result = self.client().post('/train_lr')
        self.assertEqual(result.status_code, 200)

    def test_train_br(self):
        result = self.client().post('/train_br')
        self.assertEqual(result.status_code, 200)

    def test_train_dtr(self):
        result = self.client().post('/train_dtr')
        self.assertEqual(result.status_code, 200)
        