#!/usr/bin/env bash
# Santos et al. 2024 (Data in Brief 53:110132), CC BY 4.0 — one zip per survey year from figshare
set -e
declare -A IDS=([2010]=43169008 [2015]=43169002 [2017]=43169005 [2018]=43169011 [2021]=43168999)
for y in "${!IDS[@]}"; do
  mkdir -p raw/$y
  curl -L -o raw/$y.zip "https://api.figshare.com/v2/file/download/${IDS[$y]}"
  unzip -q -o raw/$y.zip -d raw/$y      # -> raw/<year>/<year>/*.jpg + YOLO .txt labels (0 = MILCO, 1 = NOMBO)
done
