# Releases

## SkWeak v0.0.1
1. New method implemented: ...

## What's Changed
* Update README.md with ...

## Package
* makefun -> with_signature
* iteration-utilities -> deepflatten, flatten

## Pytest
* pytest -k "keyword" or pytest -k "name and not slow"	
* pytest --pdb (drop into debugger on failure)
* pytest -m "slow" (run tests with mark)
* pytest -v -s -x  (verbose, show print, and stop on fail)
* pytest.fixture
    * scope
    - function: default, run for each test
    - alternative: class, module(py file) and session
    

## Pdb
* b MyClass.predict. # breaks the next time MyClass.predict is called
* c                  # continue until it's hit
* j 120              # jump to line 120
* b my_module.py:42  # breaks at line 42 in my_module.py
* b myfile.py:87, x > 5     # breaks only if condition is true
* disable/enable/clear id (type b to show all breakpoints)
* ss -ltnp | grep 5680      # check port listening 
* sudo ufw allow 5680/tcp   # allow port passing firewall blocking
* telnet 128.175.21.77 5680 # try to connect to this port from home machine

## UV
* uv add "rlberry-research @ git+https://github.com/rlberry-py/rlberry-research.git"

## Sphinx
* cd docs/themes && git clone https://github.com/scikit-learn/scikit-learn.git temp_sklearn
