import os
import subprocess
import sys


def kill(kind):
    """Kill docker process with name=kind (e.g. vgg)."""
    pids = []
    output = subprocess.check_output("docker ps", shell=True)
    lines = output.split("\n")
    for line in lines[1:]:
        pid = line.split("        ")[0]
        print pid
        quotes = line.split("\"")
        if len(quotes) > 2:
            command = quotes[1]
            print command
            if kind in command or (
                    "inception" in command or "run_hp" in command):
                pids.append(pid)
        print pids

    for pid in pids:
        print pid
        os.system("docker kill {}".format(pid))


if __name__ == "__main__":
    kind = sys.argv[1]
    kill(kind)
