from pathlib import Path
p = Path('app/app.py')
s = p.read_text(encoding='utf-8')
stack = []
pairs = {'(':')','[':']','{':'}'}
openers = set(pairs.keys())
closers = set(pairs.values())
for i,ch in enumerate(s, start=1):
    if ch in openers:
        stack.append((ch,i))
    elif ch in closers:
        if not stack:
            print('Unmatched closer', ch, 'at', i)
            break
        last, pos = stack.pop()
        if pairs[last] != ch:
            print('Mismatched at', i, 'found', ch, 'expected', pairs[last], 'opened at', pos)
            break
else:
    if stack:
        print('Unclosed opener', stack[-1][0], 'at', stack[-1][1])
    else:
        print('All balanced')
