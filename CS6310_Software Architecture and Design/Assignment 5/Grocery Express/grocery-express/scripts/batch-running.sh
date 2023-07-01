#!/usr/bin/env bash

mkdir -p ./docker_results
exec &> "./docker_results/batch_results.txt"

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for i in {0..65}
do
  echo " ------ Test Case: $i ----"
  tc="00$i"
  "$THIS_DIR/test-running.sh" ${tc:(-2):2}
done
wait