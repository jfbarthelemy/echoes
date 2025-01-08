import os
import subprocess
import sys

def convert_qmd_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".qmd"):
                qmd_file = os.path.join(root, file)
                ipynb_file = qmd_file.replace(".qmd", ".ipynb")
                command = f"quarto convert {qmd_file} --output {ipynb_file}"
                try:
                    subprocess.run(command, check=True, shell=True)
                    print(f"Converted {qmd_file} to {ipynb_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to convert {qmd_file}: {e}")
                py_file = qmd_file.replace(".qmd", ".py")
                command = f"jupytext --to script {ipynb_file}"
                try:
                    subprocess.run(command, check=True, shell=True)
                    print(f"Converted {ipynb_file} to {py_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to convert {py_file}: {e}")

def clean_ipynb_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ipynb") or file.endswith(".py"):
                script_file = os.path.join(root, file)
                try:
                    os.remove(script_file)
                    print(f"Deleted {script_file}")
                except OSError as e:
                    print(f"Failed to delete {script_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python make_scripts.py [--convert | --clean] <folder_path>")
        sys.exit(1)

    action = sys.argv[1]
    folder_path = sys.argv[2]

    if not os.path.isdir(folder_path):
        print(f"{folder_path} is not a valid directory.")
        sys.exit(1)

    if action == "--convert":
        convert_qmd_files(folder_path)
    elif action == "--clean":
        clean_ipynb_files(folder_path)
    else:
        print("Invalid option. Use --convert to convert files or --clean to delete .ipynb files.")
        sys.exit(1)
