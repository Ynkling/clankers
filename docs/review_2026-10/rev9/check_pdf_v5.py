#!/usr/bin/env python3
"""check_pdf_v5.py -- does the PDF committed at c5e4508 show the tex committed there?

    python3 docs/review_2026-10/rev9/check_pdf_v5.py [REF]     (needs pdftotext; writes only to a temp dir)

For every tex line of body text, takes each run of 6 consecutive plain words (letters only, >= 3 letters,
outside math and commands) and looks for it in the PDF's text (pdftotext, de-hyphenated, whitespace
collapsed). Prints the share found per line and lists the lines with any run not found.
"""
import os, re, subprocess, sys, tempfile
REF = sys.argv[1] if len(sys.argv) > 1 else "c5e4508"
PREV = sys.argv[2] if len(sys.argv) > 2 else "c66cc76"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
git = lambda *a: subprocess.check_output(("git", "-C", ROOT) + a)
tex = git("show", f"{REF}:multichannel_hebbian_report_v9.tex").decode()
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "v.pdf"); open(p, "wb").write(git("show", f"{REF}:multichannel_hebbian_report_v9.pdf"))
    subprocess.check_call(["pdftotext", p, os.path.join(d, "v.txt")])
    txt = open(os.path.join(d, "v.txt"), errors="replace").read()
txt = re.sub(r"-\n(?=[a-z])", "", txt)
pdfw = " ".join(re.findall(r"[A-Za-z]{3,}", txt)).lower()

def words(line):
    line = re.sub(r"%.*", "", line)
    line = re.sub(r"\$[^$]*\$", " ", line)
    line = re.sub(r"\\(texttt|ref|label|cite|eqref|url|href|includegraphics|input)\{[^}]*\}", " ", line)
    line = re.sub(r"\\[A-Za-z]+\*?", " ", line)
    return [w.lower() for w in re.findall(r"[A-Za-z]{3,}", line)]

begin = tex.split("\n").index(next(l for l in tex.split("\n") if l.startswith("\\begin{document}")))
miss, tot, found = [], 0, 0
for n, line in enumerate(tex.split("\n"), 1):
    if n <= begin: continue
    w = words(line)
    runs = [" ".join(w[i:i + 6]) for i in range(0, max(0, len(w) - 5), 3)]
    if not runs: continue
    ok = [r in pdfw for r in runs]
    tot += len(runs); found += sum(ok)
    if not all(ok):
        miss.append((n, sum(ok), len(runs), [r for r, o in zip(runs, ok) if not o][:3]))
print(f"{REF}: {found}/{tot} six-word runs of the tex found in the PDF text ({100*found/tot:.2f}%)")
print(f"lines with a run not found: {len(miss)}")
print("(table cells, headers and two-column layouts break word order in pdftotext; misses there are expected)")
# The decisive test: runs that exist only in the new tex (changed lines) against runs only in the old tex.
old = git("show", f"{PREV}:multichannel_hebbian_report_v9.tex").decode()
def allruns(t):
    out = set()
    for line in t.split("\n"):
        w = words(line)
        out |= {" ".join(w[i:i + 6]) for i in range(0, max(0, len(w) - 5))}
    return out
rn, ro = allruns(tex), allruns(old)
only_new, only_old = rn - ro, ro - rn
fn = sum(r in pdfw for r in only_new); fo = sum(r in pdfw for r in only_old)
print(f"six-word runs only in {REF}'s tex: {len(only_new)}, found in the PDF: {fn} ({100*fn/max(1,len(only_new)):.1f}%)")
print(f"six-word runs only in {PREV}'s tex: {len(only_old)}, found in the PDF: {fo} ({100*fo/max(1,len(only_old)):.1f}%)")
for n, a, b, ex in miss:
    print(f"  tex:{n}: {a}/{b} found; e.g. {ex}")
