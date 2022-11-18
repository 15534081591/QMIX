"""init"""
from .rnn_agent import RNNAgent
from .infer_agent import InferAgent


REGISTRY = {}

REGISTRY["rnn"] = RNNAgent
REGISTRY["infer"] = InferAgent
