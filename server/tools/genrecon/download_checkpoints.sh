#!/bin/bash
# parallel ranged download (the TUM server gives ~0.25 MB/s per connection, 13.7 GB in all)
D=/workspace/stac-build/vendor/GenRecon/checkpoints; mkdir -p $D/parts; cd $D
N=32
for f in sparse_structure.pt shape_slat.pt texture_slat.pt; do
  size=$(curl -sI --max-time 30 https://kaldir.vc.cit.tum.de/genrecon/$f | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  [ -z "$size" ] && { echo "no size for $f"; exit 1; }
  if [ -f "$f" ] && [ "$(stat -c %s $f)" = "$size" ]; then echo "$f already complete"; continue; fi
  echo "$f $size bytes, $N parts"
  export size N f
  seq 0 $((N-1)) | xargs -P $N -I{} bash -c '
    i={}
    chunk=$(( (size + N - 1) / N )); start=$(( i * chunk )); end=$(( start + chunk - 1 )); [ $end -ge $size ] && end=$(( size - 1 ))
    out=parts/$f.$i; want=$(( end - start + 1 ))
    for try in 1 2 3 4 5 6; do
      have=0; [ -f $out ] && have=$(stat -c %s $out)
      [ $have -ge $want ] && break
      curl -s --max-time 7200 -r $(( start + have ))-$end https://kaldir.vc.cit.tum.de/genrecon/$f >> $out || true
    done'
  cat $(for i in $(seq 0 $((N-1))); do echo parts/$f.$i; done) > $f.tmp && mv $f.tmp $f
  echo "$f done: $(stat -c %s $f) bytes (expected $size)"
  [ "$(stat -c %s $f)" = "$size" ] && rm -f parts/$f.*
done
echo "DL DONE"
