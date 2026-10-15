# create_grpc_proto.py
import os
import subprocess
import sys


def main():
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--short-import-path"):
        print("Usage: create_grpc_proto.py <proto_file> <output_dir> [--short-import-path]")
        sys.exit(1)

    proto_file = sys.argv[1]
    output_dir = sys.argv[2]
    if len(sys.argv) == 4:
        # Keep imported proto names consistent with the C++ generator, which
        # compiles sources relative to their containing package.
        proto_input = os.path.basename(proto_file)
        output_dir = os.path.join(output_dir, os.path.dirname(proto_file))
        include_dir = os.path.dirname(proto_file)
    else:
        proto_input = proto_file
        include_dir = "."

    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{include_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            proto_input,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
