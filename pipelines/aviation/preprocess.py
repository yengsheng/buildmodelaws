import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler)