# The run's work directories leave when it ENDS WELL — sourced by run_cloud.sh.
#
# USER 2026-11-14: "borra los intermedios y procura que se eliminen cuando termina genrecon para que
# no quede basura" (29.4 GB of five finished runs sat in vendor/GenRecon/work with 14 GB of the
# volume left). $W (the adapter's inputs), $O (the generation) and $OL (the bake) are what a FAILED
# run resumes from (resume.py: "se continúa desde el punto de falla") — so they are removed only
# after the mesh is published and tiled, never on a failure's way out.
#   GENRECON_KEEP_WORK=1    keep everything (genrecon.keep_work: true — to look at a run's own files)
#   GENRECON_KEEP_INPUTS=1  keep $W only: another variant of the same session follows and reuses it
# A path is removed only when it lies strictly inside $G/work/ (an empty session id never empties
# the whole directory).
drop_work() {
  if [ "${GENRECON_KEEP_WORK:-0}" = "1" ]; then
    echo "=== work kept (genrecon.keep_work): $W $O $OL"
    return 0
  fi
  local d dirs="$O $OL"
  [ "${GENRECON_KEEP_INPUTS:-0}" = "1" ] || dirs="$W $dirs"
  for d in $dirs; do
    case "$d" in
      "$G/work/"?*) rm -rf -- "$d" && echo "=== work removed: $d" ;;
      *) echo "=== work NOT removed (outside $G/work/): $d" ;;
    esac
  done
  return 0
}
