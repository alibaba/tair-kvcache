import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="out", action="store")
    args = parser.parse_args()

    outs = args.out
    output = os.popen('/usr/bin/realpath %s' % __file__).read()

    path = os.path.abspath(os.path.join(os.path.dirname(output), '../..'))
    cmd = 'GIT_DIR=%s/.git GIT_WORK_TREE=%s git log -10 > %s' % (path, path, outs)
    os.system(cmd)
