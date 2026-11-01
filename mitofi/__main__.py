"""Lets the tool run as `python -m mitofi ...`.

pip puts console scripts in a Scripts or bin directory that is frequently not on
PATH, especially on Windows. Running the module works regardless, using whichever
Python you invoked.
"""
from .cli import main

main()
