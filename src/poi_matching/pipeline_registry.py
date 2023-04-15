"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline.modular_pipeline import Pipeline, pipeline
from kedro.pipeline import Pipeline
from .pipelines import *

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    preprocess_matching_pairs = preprocessing()

    
    return {
        'preprocess_matching_pairs': preprocess_matching_pairs
    }
