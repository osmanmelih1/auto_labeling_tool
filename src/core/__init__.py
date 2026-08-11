"""Core pipeline modules.

Each step is a standalone, decoupled stage that reads its input from and writes
its output to the ``data/`` directory hierarchy. Modules never import each other's
internals; they communicate only through files (JSON, TXT, images, NPZ).
"""
