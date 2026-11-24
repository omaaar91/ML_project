import os
import nbformat
from nbclient import NotebookClient

notebook_path = 'Notebooks/Cleanusage_behavior.ipynb'
notebook_dir = os.path.abspath('Notebooks')

print(f"Loading notebook from {notebook_path}...")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Total cells to execute: {len(nb.cells)}")
print(f"Executing with cwd: {notebook_dir}")

client = NotebookClient(
    nb, 
    timeout=600, 
    kernel_name='python3', 
    resources={'metadata': {'path': notebook_dir}}
)

print("Executing all cells...")
client.execute()

print("Execution completed! Saving executed notebook...")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Saved executed notebook successfully!")
