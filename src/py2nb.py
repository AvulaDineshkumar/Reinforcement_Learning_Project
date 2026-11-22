"""Convert a '# %%' percent-format script into a Jupyter notebook."""
import sys, nbformat
src = open(sys.argv[1]).read()
cells = []
for block in src.split("\n# %%")[1:]:
    head, _, body = block.partition("\n")
    if head.strip().startswith("[markdown]"):
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").split("\n")]
        cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, sys.argv[2])
print("wrote", sys.argv[2], len(cells), "cells")
