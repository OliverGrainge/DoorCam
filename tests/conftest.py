import sys
import os
import torch
from glob import glob
import os
import pytest
import yaml

root = "/".join(sys.path[0].split('/')[:-1]) + "/doorcam/"
sys.path.insert(0, root)
os.chdir(root)




@pytest.fixture
def config():
    print(glob("*"))
    with open("doorcam/config.yaml", "r") as file:
        return yaml.safe_load(file)