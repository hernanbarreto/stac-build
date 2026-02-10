#!/bin/bash
# Clean CRLF from all Python files in server directory
cd /home/hernan/stac-builder/server
for f in *.py; do
    sed -i 's/\r$//' "$f"
    echo "Cleaned: $f"
done
echo "Done!"
