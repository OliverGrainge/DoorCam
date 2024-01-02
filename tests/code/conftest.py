import sys
import os
import torch
from glob import glob
import os

root = "/".join(sys.path[0].split('/')[:-2]) + "/doorcam"
os.chdir(root)

import pytest

@pytest.fixture
def config():
    from doorcam.utils import get_config
    return get_config()