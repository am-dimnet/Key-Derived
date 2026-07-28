#!/bin/bash
source /etc/network_turbo 2>/dev/null
cd "${ARTICLE4_ROOT:-..}/data/raw/ptbdb" || exit 1
BASE="https://physionet.org/files/ptbdb/1.0.0"
ok=0; fail=0; n=0
total=$(wc -l < "$(dirname "$0")/ptb_missing.txt")
while read -r rec; do
  n=$((n+1)); d=$(dirname "$rec"); mkdir -p "$d"
  for ext in hea dat; do
    [ -s "${rec}.${ext}" ] && continue
    curl -sfL --retry 3 --retry-delay 2 --max-time 180 -o "${rec}.${ext}" "${BASE}/${rec}.${ext}" || rm -f "${rec}.${ext}"
  done
  if [ -s "${rec}.hea" ] && [ -s "${rec}.dat" ]; then ok=$((ok+1)); else fail=$((fail+1)); echo "FAIL $rec"; fi
  [ $((n % 20)) -eq 0 ] && echo "progress $n/$total ok=$ok fail=$fail"
done < "$(dirname "$0")/ptb_missing.txt"
echo "DONE ok=$ok fail=$fail"
