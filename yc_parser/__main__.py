#!/usr/bin/env python3
"""
Entry point for running yc_parser as a module.

Usage: python -m yc_parser [command] [options]
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())