import os
import nbformat
from nbclient import NotebookClient

notebook_path = 'Notebooks/Clean_service_transactions.ipynb'
notebook_dir = os.path.abspath('Notebooks')

print(f"Loading notebook: {notebook_path}")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': notebook_dir}})
print("Executing all cells...")
client.execute()

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Execution completed successfully! Notebook updated with outputs.")
