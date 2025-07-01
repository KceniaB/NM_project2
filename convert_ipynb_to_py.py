#%%
import nbformat

# Read your notebook (version 4)
nb = nbformat.read('12_alldatain1file.ipynb', as_version=4)

# Open an output .py
with open('12_alldatain1file.py', 'w', encoding='utf-8') as outf:
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # write each code cell, separated by blank lines
            outf.write(cell.source.rstrip() + '\n\n')
print("Done — code written to 12_alldatain1file.py")
