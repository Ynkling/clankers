"""Rebuild every video/data/*.json from results/X, results/L and the report transcription.
python3 video/data/_build/run_all.py   (no torch needed)"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for name in ("build_slow_start", "build_stream_recipe", "build_early_recipe", "build_hinge", "build_gate",
             "build_recipe_scope", "build_eight", "build_tables"):
    print(f"== {name}")
    runpy.run_path(os.path.join(HERE, name + ".py"), run_name="__main__")
