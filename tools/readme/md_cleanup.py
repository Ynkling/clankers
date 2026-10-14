"""Stage 2 of the README build: inline math to plain text where it is simple (counts, variables, bold counts), the
headings numbered as the README numbers them, tables with their captions first. Usage: python3 md_cleanup.py <workdir>."""
import re, json, sys, os
W = sys.argv[1]
md=open(os.path.join(W,'stage1.md'), encoding='utf-8').read()
SUP={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','a':'ᵃ','b':'ᵇ','c':'ᶜ','d':'ᵈ','m':'ᵐ','s':'ˢ','w':'ʷ','e':'ᵉ','h':'ʰ'}
def sup(x): return ''.join(SUP[c] for c in x)
def conv_inline(m):
    orig=m.group(0); s=m.group(1)
    t=s.replace('{=}','=').replace('\\times','×').replace('\\geq','≥').replace('\\leq','≤').replace('\\to','→')
    t=re.sub(r'^\\mathbf\{([^{}]*)\}(\^\{?[a-z]\}?)?$', lambda mm: '**'+mm.group(1)+'**'+(sup(mm.group(2).strip('^{}')) if mm.group(2) else ''), t)
    t=t.replace('\\mathbf{','').replace('{,}',',')
    t=t.replace('\\eta','η').replace('\\rho','ρ').replace('\\,',' ').replace('\\%','%').replace('\\sim','~')
    t=t.replace('\\beta','β').replace('\\alpha','α').replace('\\tanh','tanh').replace('\\pm','±').replace('\\sqrt{3}','√3')
    t=t.replace('c^*','c*').replace('c_0','c₀').replace('r_0','r₀').replace('^R','ᴿ').replace('^{f}','ᶠ')
    t=re.sub(r'\^\{(-?[0-9]+|[abcdmsweh])\}', lambda mm: sup(mm.group(1)), t)
    t=re.sub(r'\^([0-9])', lambda mm: sup(mm.group(1)), t)
    t=re.sub(r'_\{\\mathrm\{(index|half)\}\}', r'_\1', t)
    if any(c in t for c in '\\_^{}'):
        return orig
    t=re.sub(r'(?<![\w*/.])([SPkNDhdpts])(?![\w*])', r'*\1*', t)
    return t
md=md.replace('$\\mathbf{38/40}$','**38/40**')
# inline math (not $$)
md=re.sub(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', conv_inline, md)
# display math blocks
def disp(m):
    s=m.group(1); lab=re.search(r'\\label\{eq:(\w+)\}', s)
    num={'score':1,'gate':2,'window':3}[lab.group(1)] if lab else None
    s=re.sub(r'\\label\{[^}]*\}','',s).strip().rstrip(',.').strip()
    s=s.replace('\\;',' ')
    return '\n\n$$\n'+s+(f' \\qquad ({num})' if num else '')+'\n$$\n\n'
md=re.sub(r'\s*\$\$(.+?)\$\$\s*', disp, md, flags=re.S)
# quotes
md=md.replace('’',"'").replace('‘',"'").replace('“','"').replace('”','"')
# headings
out=[]; sec=0; sub=0; lines=md.split('\n'); i=0
while i<len(lines):
    L=lines[i]
    if L.startswith('# ') :
        title=L[2:].strip()
        if title=='Provenance': out.append('## Provenance')
        elif title.startswith('What a replication needs'): out.append('## '+title)
        else:
            sec+=1; sub=0; out.append(f'## {sec}. {title}')
    elif L.startswith('## '):
        sub+=1; out.append(f'### {sec}.{sub} {L[3:].strip()}')
    elif L.startswith('#### '):
        title=L[5:].strip()
        # next non-empty line joins
        j=i+1
        while j<len(lines) and lines[j].strip()=='' : j+=1
        if j<len(lines) and not lines[j].startswith(('-','|','<','#','1.','$$')):
            out.append(f'**{title}** {lines[j]}'); i=j
        else:
            out.append(f'**{title}**')
    else: out.append(L)
    i+=1
md='\n'.join(out)
# tables: move caption before table
tabs=json.load(open(os.path.join(W,'maps.json')))['tabs']
def tabblock(m):
    tid=m.group(1); inner=m.group(2).strip('\n')
    parts=[p for p in re.split(r'\n\s*\n', inner) if p.strip()]
    table=[p for p in parts if p.lstrip().startswith('|')]
    cap=[p for p in parts if not p.lstrip().startswith('|')]
    n=tabs.get(tid,'?')
    return f'**Table {n}.** '+' '.join(c.strip() for c in cap)+'\n\n'+'\n\n'.join(table)+'\n'
md=re.sub(r'<div id="(tab:[^"]+)">\n(.*?)\n</div>', tabblock, md, flags=re.S)
md=re.sub(r'\n{3,}', '\n\n', md)
open(os.path.join(W,'stage2.md'),'w',encoding='utf-8').write(md)
print(len(md.split('\n')), 'lines; remaining $ spans:', len(re.findall(r'(?<!\$)\$(?!\$)[^$\n]+?\$', md)))
