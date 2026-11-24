import os
import nbformat
from nbclient import NotebookClient

# 1. Build the notebook first
exec(open('scratch/build_part7_notebook.py', encoding='utf-8').read())

notebook_path = 'Notebooks/Part7_combine_supervised_unsupervised.ipynb'
notebook_dir = os.path.abspath('Notebooks')

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': notebook_dir}})
print("Executing all cells of Part 7...")
client.execute()

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Execution of Part 7 completed successfully! Notebook updated with outputs.")
