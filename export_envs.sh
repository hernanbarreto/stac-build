#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
mkdir -p docs/migration
for env in stac-build da3 CloudComPy310 sam3 mapanything; do
    echo "Exporting $env..."
    conda env export -n "$env" > "docs/migration/environment_$env.yml"
done
echo "Done"
