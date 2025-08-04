from confection import Config
from unittest import mock


def mock_config_from_str(config_str, *, interpolate = True, overrides={}):
    import re
    # Pattern 1: {key: value} -> dict(key=value)
    def replace_dict_1(text):
        def convert_brace_dict(match):
            content = match.group(1)
            content = re.sub(r"\s+", " ", content).strip()
            # Skip if it's just numbers (like {1,2,3})
            if re.match(r"^[\d,\s]+$", content):
                return match.group(0)
            content = re.sub(r"(\w+):\s*(\w+)", r"\1=\2", content)
            return f"dict({content})"

        return re.sub(
            r"\{([^{}]*(?:\n[^{}]*)*)\}",
            convert_brace_dict,
            text,
            flags=re.DOTALL
        )
    # Pattern 2: \n key = value\n key = value -> dict(key=value, key=value)
    def replace_dict_2(text):
        def convert_assignments(match):
            prefix, content = match.groups()  
            assignments = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if "=" in line and not line.startswith("{"):
                    k, v = line.split("=", 1)
                    assignments.append(f"{k}={v}")
            if assignments:
                return f"{prefix}: dict({', '.join(assignments)})\n"
            return match.group(0)

        return re.sub(
            r"(.*):\s*\n((?:\s*(?:.*=.*)?\n)*?)(?=^\S|\Z)",
            convert_assignments,
            text,
            flags=re.MULTILINE,
        )   # start with indentation, contain at least one x=y followed by \n?
        
    # Pattern 3: [ \n 1, \n 2, ... ] -> [1, 2, ...]
    def replace_list_1(text):
        def convert_assignments(match):
            return "[" + re.sub(r"\n", "", match.group(1)) + "]"
        
        return re.sub(
            r"\[(.*?)\]",
            convert_assignments,
            text,
            flags=re.DOTALL,
        )
        
    # eval dict() ... in nested dictionary
    def safe_eval(s):
        s = re.sub(r"=\s*([a-zA-Z_]\w*)", r'="\1"', s)
        return eval(s)
        
    def parse_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = parse_dict(v)
            elif isinstance(v, str) and v.startswith('dict('):
                d[k] = safe_eval(v)
            elif isinstance(v, str) and v.startswith('{'):
                d[k] = safe_eval(v)
        return d
    
    config_str = replace_dict_1(config_str)
    config_str = replace_dict_2(config_str)
    config_str = replace_list_1(config_str)

    cdic = original_Config_fromstr(Config(), config_str, interpolate=interpolate, overrides=overrides)
    rtn = {**cdic.pop("root"), **cdic}
    return parse_dict(rtn)


original_Config_fromstr = Config.from_str
mock.patch("confection.Config.from_str", side_effect=mock_config_from_str).start()


y = """
[root]
x = 1
func_kwargs: 
    name = mushroom
    x = 1
    

[data]

[data.trec]
func = WeakTextData.from_name
y1 = {a: 1, b: 2}
y2 = {1,2,3}
x1:
    a = 1
    b = 2
x2: {
    a: 1,
    b: 2
}
x3:
    a = 1
    
[data.p]
z = [
    1,
    2
]


[data.mushroom]
func = WeakData.from_name

[agent]
"""
x = Config().from_str(y)

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    print(x)