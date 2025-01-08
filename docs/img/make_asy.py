import os
import subprocess
import sys

def generate_outputs(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".asy"):
                asy_file = os.path.join(root, file)
                html_output = asy_file.replace(".asy", ".html")
                pdf_output = asy_file.replace(".asy", ".pdf")
                try:
                    subprocess.run(f"asy -f html --nopdfreload --noView {asy_file}", check=True, shell=True)
                    # subprocess.run(f"asy -f pdf --nopdfreload --noView {asy_file}", check=True, shell=True)
                    # print(f"Generated {html_output} and {pdf_output} from {asy_file}")
                    print(f"Generated {html_output} from {asy_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to generate outputs for {asy_file}: {e}")

def clean_outputs(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".html") or file.endswith(".pdf"):
                output_file = os.path.join(root, file)
                try:
                    os.remove(output_file)
                    print(f"Deleted {output_file}")
                except OSError as e:
                    print(f"Failed to delete {output_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_asy.py [--generate | --clean] <folder_path>")
        sys.exit(1)

    action = sys.argv[1]
    folder_path = sys.argv[2]

    if not os.path.isdir(folder_path):
        print(f"{folder_path} is not a valid directory.")
        sys.exit(1)

    if action == "--generate":
        generate_outputs(folder_path)
    elif action == "--clean":
        clean_outputs(folder_path)
    else:
        print("Invalid option. Use --generate to create outputs or --clean to delete HTML and PDF files.")
        sys.exit(1)
