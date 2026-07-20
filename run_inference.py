import argparse
from do_infer import main as do_infer

def main():
    parser = argparse.ArgumentParser(description="Run nnUNetv2 inference")
    parser.add_argument("--input", type=str, default="/inputs", help="Path to input directory")
    parser.add_argument("--output", type=str, default="/outputs", help="Path to output directory")
    args = parser.parse_args()
    do_infer(args.input, args.output, verbose = True)
    print("ok")

if __name__ == "__main__":
    main()