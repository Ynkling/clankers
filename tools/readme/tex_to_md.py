"""Stage 1 of the README build: the report's body (from the revision paragraphs to the bibliography) through pandoc,
with the citations numbered as the References section numbers them and the section, table and equation references
resolved to the README's anchors. Usage: python3 tex_to_md.py <report.tex> <workdir>."""
import re, subprocess, sys, os
TEX, W = sys.argv[1], sys.argv[2]
os.makedirs(W, exist_ok=True)
tex=open(TEX, encoding='utf-8').read()
body=tex[tex.index('\\paragraph{Revision~8'):tex.index('\\begin{thebibliography}')]
# citations
cites={'bdh':1,'bdhcq':2,'tla':3,'mamba':4,'ttr':5,'gdn':6,'trm':7,'recdepth':8,'flesch':9,'dendrites':10,'cgar':11,'mom':12,'raven':13,'routing':14,'hmrnn':15,'nested':16,'switch':17,'zoology':18,'bietti':19,'edelman':20}
body=re.sub(r'\s*\\cite\{([^}]*)\}', lambda m:' ['+', '.join(str(cites[c.strip()]) for c in m.group(1).split(','))+']', body)
body=re.sub(r'\\arm\{([^}]*)\}', r'\\texttt{\1}', body)
# pandoc drops a number that begins a cell of a >{\raggedright\arraybackslash}p{..} column; an empty group in front of it
# keeps it (verified on pandoc 3: '& 140 of 156' -> 'of 156', '& {}140 of 156' -> '140 of 156')
def _guard_cells(block):
    block = re.sub(r'(&\s*)(?=\d)', r'\1{}', block)
    return re.sub(r'^(\s*)(?=\d)', r'\1{}', block, flags=re.M)
body = re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})', lambda m: _guard_cells(m.group(1)), body, flags=re.S)
# section map
secs={}; n=0; sub=0; app=False; n_app=0
for m in re.finditer(r'\\(section|subsection|appendix)(?:\{([^}]*)\})?(?:\s*\\label\{([^}]*)\})?', body):
    kind,title,label=m.groups()
    if kind=='appendix': app=True; continue
    if kind=='section':
        n+=1; sub=0
        num=('A' if n_app==0 else 'B') if app else str(n)
        if app: n_app+=1
        if label: secs[label]=(num,title)
    else:
        sub+=1
        if label: secs[label]=(f'{n}.{sub}',title)
secs['sec:main']=('5','The main result: test_window_gate')
tabs={}; t=0
for m in re.finditer(r'\\label\{(tab:[^}]*)\}', body):
    t+=1; tabs[m.group(1)]=t
eqs={'eq:score':1,'eq:gate':2,'eq:window':3}
open(os.path.join(W,'body.tex'),'w',encoding='utf-8').write('\\documentclass{article}\\begin{document}\n'+body+'\n\\end{document}')
md=subprocess.check_output(['pandoc',os.path.join(W,'body.tex'),'-f','latex','-t','gfm+tex_math_dollars','--wrap=none'],text=True)
def slug(s):
    s=s.lower(); s=re.sub(r'[^\w\- ]','',s); return s.replace(' ','-')
def anchor(num,title):
    t=title.replace('~',' ')
    t=re.sub(r'\\[a-z]+\{?','',t).replace('}','')
    return slug(f'{num}. {t}' if '.' not in num else f'{num} {t}') if num!='A' else 'provenance'
def ref(m):
    lab=m.group(1)
    if lab in secs:
        num,title=secs[lab]
        if lab=='app:prov': return '[Provenance](#provenance)'
        if lab=='app:repro': return '[the replication protocol](#what-a-replication-needs-the-protocol-as-this-revision-states-it)'
        return f'[{num}](#{anchor(num,title)})'
    if lab in tabs: return str(tabs[lab])
    if lab in eqs: return f'({eqs[lab]})'
    return '??'+lab
md=re.sub(r'<a href="#([^"]+)" data-reference-type="ref" data-reference="[^"]+">[^<]*</a>', ref, md)
md=md.replace('Appendix [Provenance](#provenance)','the [Provenance](#provenance) section')
md=md.replace('equation~(','equation (').replace('equation (3)','equation (3)')
open(os.path.join(W,'stage1.md'),'w',encoding='utf-8').write(md)
import json; json.dump({'secs':secs,'tabs':tabs},open(os.path.join(W,'maps.json'),'w'),indent=1)
print(secs); print(tabs)
