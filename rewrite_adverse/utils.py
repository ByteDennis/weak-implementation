import re
from unittest import mock
from confection import Config


#   ┌────────────────────────────────────────────────────────────────────────────┐
#   │ original_Config_fromstr = Config.from_str                                  │
#   └────────────────────────────────────────────────────────────────────────────┘
original_Config_fromstr = Config.from_str
def mock_config_from_str(config_str):
    # Remove newlines within section headers to match your preprocessing
    config_str = re.sub(
        r"\[(.*?)\]",
        lambda m: "[" + re.sub(r"\n", "", m.group(1)) + "]",
        config_str,
        flags=re.DOTALL,
    )
    # Use original Config().from_str to parse the preprocessed string
    cdic = original_Config_fromstr(Config(), config_str)
    return {**cdic.pop("root"), **cdic}
mock.patch("confection.Config.from_str", side_effect=mock_config_from_str).start()



