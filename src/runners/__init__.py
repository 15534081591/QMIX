"""runner init"""
from .episode_runner import EpisodeRunner


REGISTRY = {}

REGISTRY["episode"] = EpisodeRunner
