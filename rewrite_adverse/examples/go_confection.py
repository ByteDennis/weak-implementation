import re
from unittest import mock
from confection import Config


original_Config_fromstr = Config.from_str
def mock_config_from_str(config_str):
    # Remove newlines within section headers to match your preprocessing
    config_str = re.sub(
       r"\[(.*?)\]", 
       lambda m: "[" + re.sub(r"\n", "", m.group(1)) + "]", 
       config_str, 
       flags=re.DOTALL
   )
    # Use original Config().from_str to parse the preprocessed string
    cdic = original_Config_fromstr(Config(), config_str)
    return {**cdic.pop("root"), **cdic}

mock.patch('confection.Config.from_str', side_effect=mock_config_from_str).start()


CONFIG_STR = """
[model]
n_epochs = 150
lr = 0.005

[data]
data_dir = "./an2024convergence/datasets/"
data_name = [
    "imdb",
    "aa2"
]
use_test = true

[root]
run_name = "hyperlm_run"
"""

c = Config().from_str(CONFIG_STR)
print(c)
