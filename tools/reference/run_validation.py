"""Run the complete numerical reference gate with lossless subprocess logs."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,default=Path('.artifacts'))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts = a.artifacts.resolve()
    assets, reference, logs = artifacts/'assets', artifacts/'reference', artifacts/'task2-logs/raw'
    logs.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, FT_SOURCE_DIR=str(a.source.resolve()), FT_SUPPLIED_MODELS_DIR=str(a.models.resolve()), FT_ASSETS_DIR=str(assets), FT_REFERENCE_DIR=str(reference))
    commands = [
        ['tools/reference/audit.py','--source',str(a.source.resolve()),'--models',str(a.models.resolve()),'--output',str(artifacts/'audit.json')],
        ['tools/reference/convert.py','--source',str(a.source.resolve()),'--models',str(a.models.resolve()),'--output',str(assets)],
        ['tools/reference/generate.py','--source',str(a.source.resolve()),'--assets',str(assets),'--output',str(reference)],
        ['tools/reference/verify.py','--reference',str(reference)],
        ['-m','pytest','tools/reference/tests/test_assets.py','-q'],
    ]
    results = []
    for number, command in enumerate(commands):
        result = subprocess.run([sys.executable,*command],cwd=root,env=env,capture_output=True)
        stdout,stderr = logs/f'{number}-stdout.log',logs/f'{number}-stderr.log'
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        results.append({'argv':[sys.executable,*command],'cwd':str(root),'returncode':result.returncode,'stdout':str(stdout),'stderr':str(stderr)})
        (logs/'commands.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
        print(f'{number}: exit={result.returncode}, stdout={stdout.name}, stderr={stderr.name}',flush=True)
        if result.returncode:
            sys.stdout.buffer.write(result.stdout)
            sys.stderr.buffer.write(result.stderr)
            return result.returncode
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
