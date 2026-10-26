#!/usr/bin/env python3
"""kernel_slicer test system. Run without arguments for an interactive menu, '--help' for options.
Implementation: python_scripts/ktest/; documentation: python_scripts/ktest/README.md"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'python_scripts'))

from ktest.cli import main

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:], ROOT))
