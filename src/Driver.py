import sys

sys.path.append('../')

from src.models.DNNModel import DNNModel

m = DNNModel()
m.load_data()
m.train()
m.create_predictions()