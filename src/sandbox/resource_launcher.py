import os
import resource
import sys


def run_generated_script_with_limits() -> None:
    """Apply resource limits before executing the generated Python script."""

    cpu_seconds = int(sys.argv[1])
    max_file_bytes = int(sys.argv[2])
    script_path = sys.argv[3]

    if cpu_seconds > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))

    if max_file_bytes > 0:
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))

    os.execv(sys.executable, [sys.executable, script_path])


if __name__ == "__main__":
    run_generated_script_with_limits()


