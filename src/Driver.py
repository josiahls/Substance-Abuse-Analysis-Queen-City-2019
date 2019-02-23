import sys

sys.path.append('../')

from src.models.DNNModel import DNNModel

m = DNNModel()
m.train()
