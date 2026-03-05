import ast, sys

files = [
    '/root/claude-projects/whisper-voice-public/main.py',
    '/root/claude-projects/whisper-voice-public/build.py',
    '/root/claude-projects/whisper-voice-public/src/app.py',
]
ok = True
for f in files:
    with open(f) as fh:
        src = fh.read()
    try:
        ast.parse(src)
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'SYNTAX ERROR in {f}: {e}')
        ok = False

sys.exit(0 if ok else 1)
