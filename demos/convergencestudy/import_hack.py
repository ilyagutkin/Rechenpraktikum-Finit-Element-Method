"""
This appends the '../../src' directory to the system path, allowing modules from that directory to be imported.
This allows to use updated versions of the modules without having to reinstall them.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
import methodsnm