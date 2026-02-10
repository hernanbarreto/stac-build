#!/bin/bash
# Fix truncated text in Python files
cd /home/hernan/stac-builder/server

echo "Fixing truncated text in Python files..."

# Common truncations to fix (word -> correction)
declare -A fixes=(
    ["asynccontextmanage"]="asynccontextmanager"
    ["AlignmentManage"]="AlignmentManager"
    ["FrameTracke"]="FrameTracker"
    ["FactorGrap"]="FactorGraph"
    ["RetrievalDatabas"]="RetrievalDatabase"
    [": st$"]="str"
    [": flo$"]="float"
    [": in$"]="int"
    [": boo$"]="bool"
    [": lis$"]="list"
    [": dic$"]="dict"
    [": Optiona$"]="Optional"
    [": np.ndarr$"]="np.ndarray"
    ["Wrappe$"]="Wrapper"
    ["# mast3$"]="# mast3r"
    ["# dust3$"]="# dust3r"
)

for f in *.py; do
    for pattern in "${!fixes[@]}"; do
        sed -i "s/${pattern}/${fixes[$pattern]}/g" "$f"
    done
done

# Also fix any line endings
for f in *.py; do
    sed -i 's/\r$//' "$f"
done

echo "All fixes applied!"

# Verify by trying to import main
echo ""
echo "Testing import of main.py..."
cd /home/hernan/stac-builder/server
/home/hernan/miniforge3/envs/mast3r-slam/bin/python -c "
import sys
sys.path.insert(0, '/home/hernan/stac-builder/server')
try:
    import main
    print('SUCCESS: main.py imports correctly!')
except Exception as e:
    print(f'ERROR: {e}')
"
