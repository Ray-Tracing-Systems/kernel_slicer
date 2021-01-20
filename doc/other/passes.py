#hello.py - http://www.graphviz.org/content/hello
from graphviz import Digraph

pass0 = "#0. The very first pass";
pass1 = "#1. Initial Pass";
pass2 = "#2. Control Function (CF) Process";
pass3 = "#3. Kernel Functions (KF) Process";
pass4 = "#4. Definitions Extracting from Main Class";
pass5 = "#5. Control Function (CF) Rewrite";
pass6 = "#6. Offsets for data members";
pass7 = "#7. Host code Rendering (Vulkan calls)";
pass8 = "#8. Kernel Functions (KF) Rewrite";

g = Digraph('G', filename='hello.gv')

g.edge(pass0, pass1)
g.edge(pass1, pass2)
g.edge(pass2, pass3)
g.edge(pass3, pass4)
g.edge(pass4, pass5)
g.edge(pass5, pass6)
g.edge(pass6, pass7)
g.edge(pass7, pass8)

g.edge(pass2, pass5)
g.edge(pass3, pass8)

g.view()
