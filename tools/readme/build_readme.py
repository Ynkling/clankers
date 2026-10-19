#!/usr/bin/env python3
"""Build README.md from the technical report's tex and the hand-written parts in parts/.

    python3 tools/readme/build_readme.py [multichannel_hebbian_report_v10.tex]

Stages: tex_to_md.py (pandoc, references resolved) -> md_cleanup.py (inline math, headings, tables) -> this file,
which assembles header + Summary + Status + the report's sections + the repository layout + the appendices + the
references + Pathway's upstream README, and regenerates the Contents list from the headings. The report's sections,
Table 1 and the Revision paragraph are generated from the tex; parts/summary.md, parts/status.md (its fixed text),
parts/repo_layout.md and parts/references.md are written by hand and must be brought up to date with each revision.
Needs pandoc. The work directory (tools/readme/work/, gitignored) holds the intermediate files.
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'multichannel_hebbian_report_v10.tex')
W = os.path.join(HERE, 'work')
P = os.path.join(HERE, 'parts')

subprocess.check_call([sys.executable, os.path.join(HERE, 'tex_to_md.py'), TEX, W], stdout=subprocess.DEVNULL)
subprocess.check_call([sys.executable, os.path.join(HERE, 'md_cleanup.py'), W], stdout=subprocess.DEVNULL)

def part(name):
    return open(os.path.join(P, name), encoding='utf-8').read()

st2 = open(os.path.join(W, 'stage2.md'), encoding='utf-8').read()
lines = st2.split('\n')
# the Revision paragraphs come first in the body: Revision 9's and Revision 10's
revs = [l for l in lines[:14] if re.match(r'^\*\*Revision.(9|10)', l)]
assert len(revs) == 2, revs
rev_paras = '\n\n'.join(revs)
# Table 1 block: from '**Table 1.**' to the end of its table
i = [k for k, l in enumerate(lines) if l.startswith('**Table 1.**')][0]
j = i + 1
while j < len(lines) and (lines[j].startswith('|') or lines[j].strip() == ''):
    j += 1
table1 = '\n'.join(lines[i:j]).rstrip()
body = st2[st2.index('## 1. Background'):]
k = body.index('## Provenance')
body_main = body[:k].rstrip() + '\n'
appendix = body[k:].rstrip() + '\n'

status = part('status.md').replace('REVISION_PARAGRAPHS', rev_paras).replace('TABLE1', table1).rstrip() + '\n'
summary = part('summary.md').rstrip() + '\n'
repo = part('repo_layout.md').rstrip() + '\n'
refs = part('references.md')
upstream = part('upstream_bdh_readme.md')
header = part('header.md')

heads = re.findall(r'^## (.+)$', body_main + appendix, flags=re.M)
def slug(t):
    t = re.sub(r'[`*]', '', t).lower(); t = re.sub(r'[^\w\- ]', '', t); return t.replace(' ', '-')
contents = ['- [Summary](#summary)', '- [Status, machines and the Phase VI tests](#status-machines-and-the-phase-vi-tests)']
for h in heads:
    if h == 'Provenance':
        contents.append('- [Repository layout and how to run the tests](#repository-layout-and-how-to-run-the-tests)')
    contents.append(f'- [{h}](#{slug(h)})')
contents.append('- [References](#references)')
contents.append('- [About BDH (upstream README from Pathway)](#about-bdh-upstream-readme-from-pathway)')

out = header + '## Contents\n\n' + '\n'.join(contents) + '\n\n---\n\n' + summary + '\n' + status + '\n' + body_main + '\n' + repo + '\n' + appendix + '\n' + refs + upstream
open(os.path.join(ROOT, 'README.md'), 'w', encoding='utf-8').write(out)
print(len(out.split('\n')), 'lines ->', os.path.join(ROOT, 'README.md'))
