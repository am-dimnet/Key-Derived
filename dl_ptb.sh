#!/bin/bash
cd "${ARTICLE4_ROOT:-..}/data/raw/ptbdb"
BASE="https://physionet.org/files/ptbdb/1.0.0"
ok=0; fail=0
while read -r rec; do
  d=$(dirname "$rec"); mkdir -p "$d"
  for ext in hea dat xyz; do
    if [ ! -f "${rec}.${ext}" ]; then
      curl -sfL --max-time 120 -o "${rec}.${ext}" "${BASE}/${rec}.${ext}" || rm -f "${rec}.${ext}"
    fi
  done
  if [ -f "${rec}.hea" ] && [ -f "${rec}.dat" ]; then ok=$((ok+1)); else fail=$((fail+1)); echo "FAIL $rec"; fi
done < /tmp/missing.txt
echo "DONE ok=$ok fail=$fail"
