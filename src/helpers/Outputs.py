import os

import pandas as pd
from src.process.PostProcessingProcess import PostProcessingProcess


def _union_results(log, output):
    p = PostProcessingProcess(log=log, output=output)
    return p.union_results(run_tag=None, build_target=True)
