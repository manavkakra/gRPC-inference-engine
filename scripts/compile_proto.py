"""
scripts/compile_proto.py
─────────────────────────
Compiles the .proto file to Python stubs using grpc_tools.

Run once after install:
    python scripts/compile_proto.py
"""

import os
import subprocess
import sys


def compile_proto() -> None:
    proto_file  = "proto/inference.proto"
    output_dir  = "."

    if not os.path.exists(proto_file):
        print(f"ERROR: {proto_file} not found. Run from the project root.")
        sys.exit(1)

    try:
        from grpc_tools import protoc
    except ImportError:
        print("ERROR: grpcio-tools not installed. Run:")
        print("       pip install grpcio-tools")
        sys.exit(1)

    ret = protoc.main([
        "grpc_tools.protoc",
        f"-I.",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file,
    ])

    if ret != 0:
        print("Proto compilation failed.")
        sys.exit(ret)

    print("✓ Generated: proto/inference_pb2.py")
    print("✓ Generated: proto/inference_pb2_grpc.py")
    print("\nCopy or symlink these to your project root:")
    print("  cp proto/inference_pb2.py inference_pb2.py")
    print("  cp proto/inference_pb2_grpc.py inference_pb2_grpc.py")


if __name__ == "__main__":
    compile_proto()
