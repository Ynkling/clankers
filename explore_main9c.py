#!/usr/bin/env python
"""
explore_main9c.py — EXPLORATORY, not a result. Batch 10 helper: run code against the main line's
module tree at commit MAIN_SHA (9c5939e, test_slow_start's commit, whose recorded HINGE_D8 runs S33
pairs with), READ-ONLY, in a child process.

Why a child process: main's test_stream_curriculum builds BindTask(..., s_active=...) at import, a knob
only main's test_binding_capacity has; the branch's test_binding_capacity (and test_short_conv,
test_curriculum_confirm) differ from main's. Inside a child, an import hook serves every test module
that differs between this branch and MAIN_SHA (SERVED) from `git show MAIN_SHA:<file>`; every other
module (bdh.py and the unchanged test modules, explore_* helpers) is this branch's file, identical to
main's at MAIN_SHA (the CHECK lists the differences). No file is added to or changed on this branch.
The parent process never imports a served module, so the branch's own screens and the repro check run
on the branch's code.

run_child(module, func, payload) runs `module.func(payload)` in a fresh `python` with the hook
installed first and returns its JSON-serialisable result; the child's stdout is passed through.
"""

import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_REF = "claude/bdh-growth-hebbian-inference-w90069"
MAIN_SHA = "9c5939ef8ca8db079c87beb489b7c603169c9c57"
RESULTS_SHA = "b8c6007669bc2bfe08acccc3779e19ba4745685e"      # results/X/slow_start_results.json (HINGE_D8)
RECIPE_SHA = "520fe510cace8e4f698c1dfd390438281163e6df"       # results/X/stream_recipe_results.json (A4k4)
SERVED = ("test_binding_capacity", "test_curriculum_confirm", "test_short_conv", "test_scale_axes",
          "test_slow_start", "test_stream_channels", "test_stream_curriculum", "test_stream_recipe")


def _git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout


def ensure_main():
    if subprocess.run(["git", "-C", HERE, "cat-file", "-e", f"{MAIN_SHA}^{{commit}}"], capture_output=True).returncode:
        subprocess.run(["git", "-C", HERE, "fetch", "-q", "origin", MAIN_REF], check=True)


def main_blob(path, sha=MAIN_SHA):
    ensure_main()
    return _git("show", f"{sha}:{path}")


def differing():
    """Test modules (and any other non-explore .py) that differ between this branch's HEAD and MAIN_SHA."""
    ensure_main()
    names = _git("diff", "--name-only", "HEAD", MAIN_SHA, "--", "*.py").split()
    return sorted(n[:-3] for n in names if not n.startswith("explore_"))


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        src = main_blob(module.__name__ + ".py")
        module.__file__ = os.path.join(HERE, module.__name__ + ".py")
        module.__main_sha__ = MAIN_SHA
        exec(compile(src, f"<{MAIN_SHA[:7]}:{module.__name__}.py>", "exec"), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in SERVED:
            return importlib.util.spec_from_loader(name, _Loader(), origin=f"git {MAIN_SHA[:7]}:{name}.py")
        return None


def install():
    """In the child, before anything imports a test module."""
    clash = [n for n in SERVED if n in sys.modules]
    assert not clash, f"already imported from the branch: {clash}"
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


_CHILD = """
import json, sys
sys.path.insert(0, {here!r})
import torch
torch.set_num_threads(1)
import explore_main9c as mt
mt.install()
import importlib
mod = importlib.import_module({module!r})
with open({inp!r}) as f:
    payload = json.load(f)
out = getattr(mod, {func!r})(payload)
with open({outp!r}, "w") as f:
    json.dump(out, f)
"""


def run_child(module, func, payload, timeout=None):
    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "in.json"), os.path.join(td, "out.json")
        with open(inp, "w") as f:
            json.dump(payload, f)
        code = _CHILD.format(here=HERE, module=module, func=func, inp=inp, outp=outp)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(f"child {module}.{func} failed (exit {r.returncode})")
        with open(outp) as f:
            return json.load(f)


def recorded(name, sha):
    return json.loads(main_blob(f"results/X/{name}_results.json", sha))["runs"]
