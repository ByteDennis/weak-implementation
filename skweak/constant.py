
import os
import inspect
import tomllib
import subprocess

from upath import UPath
from enum import Enum
from typing import List, Any, Dict
from .logging import logger

assert "DATA" in os.environ, (
    "Please set environment variable 'DATA' (e.g., export DATA=/path/to/data)"
)

_DEBUG = False
ABSTAIN = -1
TIME_FORMAT = "%Y%m%d %H:%M:%S"
DATA_HOME   = UPath(os.environ['DATA'])

class TaskType(Enum):
    binary = 'Binary'
    multiclass = 'Multiclass'
    regression = 'Regression'

class DataType(Enum):
    tabular = 'Tabular'
    image = 'Image'
    text = 'Text'
    textrl = "TextRelation"


DATA_SET = tomllib.loads(
    """
[mushroom]
directory = "{data_dir}/OPENML/mushroom"
data_type = "{tabular}"
task_type = "{binary}"

[mushroom.meta_info]
label = {{"0" = "e", "1" = "p"}}

[cdr]
directory = "{data_dir}/classification/cdr"
data_type = "{textrl}"
task_type = "{binary}"

[cdr.meta_info]
label = {{"0" = "No", "1" = "Yes"}}
description = "Does the drug induces the disease"

[trec]
directory = "{data_dir}/classification/trec"
data_type = "{text}"
task_type = "{multiclass}"

[trec.meta_info]
label = {{"0" = "Description(DESC)", "1" = "Entity(ENTY)", "2" = "Human(HUM)", "3" = "Abbreviation(ABBR)", "4" = "Location(LOC)", "5" = "Number(NUM)"}}
description = "whether a person earns more than 50K"
                         
[census]
directory = "{data_dir}/classification/census"
data_type = "{tabular}"
task_type = "{binary}"

[census.meta_info]
label = {{"0" = "No", "1" = "Yes"}}
description = "whether a person earns more than 50K"

[agnews]
directory = "{data_dir}/classification/agnews"
data_type = "{text}"
task_type = "{multiclass}"

[agnews.meta_info]
label = {{"0" = "World", "1" = "Sports", "2" = "Business", "3" = "Sci/Tech"}}
description = "whether a person earns more than 50K"

[imdb]
directory = "{data_dir}/classification/imdb"
data_type = "{text}"
task_type = "{binary}"

[imdb.meta_info]
label = {{"0" = "HAM", "1" = "SPAM" }}
description = "whether an IMDB review is positive"

[youtube]
directory = "{data_dir}/classification/youtube"
data_type = "{text}"
task_type = "{binary}"

[youtube.meta_info]
label = {{"0" = "HAM", "1" = "SPAM" }}
description = "whether a youtube video is a spam"

[sms]
directory = "{data_dir}/classification/sms"
data_type = "{text}"
task_type = "{binary}"

[sms.meta_info]
label = {{"0" = "HAM", "1" = "SPAM" }}
description = "whether a SMS message is a spam"

[tennis]
directory = "{data_dir}/classification/tennis"
data_type = "{image}"
task_type = "{binary}"

[tennis.meta_info]
label = {{"0" = "Not Rally", "1" = "Rally" }}
description = "whether the ball is actively in play between two players, 1 frame per second (1 FPS - 2048 features) where is the image !!FIXME"
""".format(
        data_dir=DATA_HOME.as_posix(),
        tabular=DataType.tabular.value,
        binary=TaskType.binary.value,
        text=DataType.text.value,
        multiclass=TaskType.multiclass.value,
        image=DataType.image,
        textrl=DataType.textrl,
    )
)


""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Utility Functions                                                        │
  └──────────────────────────────────────────────────────────────────────────┘
"""
def get_arg_list(func) -> List[str]:
   """Return list of parameter names from a callable"""
   sig = inspect.signature(func)
   return [name for name in sig.parameters.keys() if name not in ('self', 'cls')]
   
def get_arg_dict(func) -> Dict[str, Any]:
   """Return dict with parameter names and their default values """
   sig = inspect.signature(func)
   result = {}
   for name, param in sig.parameters.items():
       if name in ('self', 'cls'):
           continue
       if param.default == inspect.Parameter.empty:
           result[name] = None  # Required parameter
       else:
           result[name] = param.default  # Optional parameter with default
   return result

def fetch_first(iterable):
    try: 
        return next(iter(iterable))
    except (StopIteration, TypeError):
        return None


def _kill_port_process(port):
    result = subprocess.run(
        f"lsof -ti:{port}", shell=True, capture_output=True, text=True
    )
    if result.stdout:
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                subprocess.run(f"kill -9 {pid}", shell=True)
                print(f"Killed process {pid} on port {port}")
    
def set_pdb(enable=True, port=5680, host="0.0.0.0", wait_for_attach=True):
    import debugpy

    global _DEBUG
    if enable:
        try:
            if not _DEBUG:
                _kill_port_process(port)
                debugpy.listen((host, port))
                _DEBUG = True
                print(f"Started debugpy listener on {host}:{port}")
        except RuntimeError as e:
            logger.error(f"debugpy RuntimeError: {e}")
        except ImportError:
            logger.error("debugpy not installed. Install with: pip install debugpy")
        except Exception as e:
            logger.error(f"Failed to enable debugging: {e}")
        if wait_for_attach:
            debugpy.wait_for_client()
