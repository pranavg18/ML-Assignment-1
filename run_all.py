"""Train both problems concurrently on CPU; stop both cleanly if interrupted"""
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys


def main():
    Path('logs').mkdir(exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    jobs = []
    with ExitStack() as stack:
        try:
            for problem in [1, 2]:
                log = Path(f'logs/train_var{problem}.log')
                stream = stack.enter_context(log.open('w'))
                print(f'Starting var{problem}; progress: {log}', flush=True)
                process = subprocess.Popen([sys.executable, '-u', 'train.py', '--problems', str(problem)],
                                           env=env, stdout=stream, stderr=subprocess.STDOUT)
                jobs.append((problem, process))
            for problem, process in jobs:
                status = process.wait()
                if status:
                    raise RuntimeError(f'var{problem} failed (exit {status}); inspect its log.')
                print(f'var{problem} completed.', flush=True)
        finally:
            for _, process in jobs:
                if process.poll() is None:
                    process.terminate()
            for _, process in jobs:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
