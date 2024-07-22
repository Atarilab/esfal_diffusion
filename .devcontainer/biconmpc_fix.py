import os

### REMOVE LINES THAT PRINTOUT SOME VALUES IN C++ FILES

FILE_PATH = [
    "/biconvex_mpc/src/motion_planner/biconvex.cpp",
    "/biconvex_mpc/src/motion_planner/kino_dyn.cpp",
]
    
LINES_TO_REMOVE = [
    "Maximum iterations reached",
    "Initialized Kino-Dyn planner",
]
    
for path in FILE_PATH:
    if os.path.exists(path):
        
        # Remove lines to remove
        with open(path, "r") as f:
            filtered_lines = [
                s
                for s in f.readlines()
                if not any(sub in s for sub in LINES_TO_REMOVE)
            ]
        
        # Override without the line removed
        with open(path, "w") as f:
            f.writelines(filtered_lines)