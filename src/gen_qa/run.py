import os
import re
import pickle
import time
import random
import threading
import json
import hashlib

import concurrent.futures
from tqdm.auto import tqdm
from openai import OpenAI
from langchain_core.documents import Document


random.seed(42)

MINIMAL_CHUNK_SIZE = 100
MAX_WORKERS = 20
INPUT_PATH = './data/processed_docs/clean_docs.pkl'