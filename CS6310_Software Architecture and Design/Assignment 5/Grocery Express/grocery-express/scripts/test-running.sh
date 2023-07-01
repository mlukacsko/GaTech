#!/usr/bin/env bash

DOCKER_COMPOSE_PREFIX="grocery-express"
DB_CONTAINER="${DOCKER_COMPOSE_PREFIX}-db-1"
CLI_CONTAINER="${DOCKER_COMPOSE_PREFIX}-cli-1"

SCENARIO=$1
OUT_DIR="./docker_results"
TEST_COMMANDS_FILE="commands_${SCENARIO}.txt"
CORRECT_RESULTS_FILE="drone_delivery_initial_${SCENARIO}_results.txt"
ACTUAL_RESULTS_FILE="drone_delivery_${SCENARIO}_results.txt"
DIFFERENCE_FILE="diff_results_${SCENARIO}.txt"

mkdir -p ${OUT_DIR}

if [[  -f "./app/src/test/cli/inputs/${TEST_COMMANDS_FILE}" ]]; then
  # truncate the database
  docker exec \
    ${DB_CONTAINER} \
    psql -U postgres -f /docker-entrypoint-initdb.d/01_truncate_tables.sql

  # repopulate initial data
  docker exec \
    ${DB_CONTAINER} \
    psql -U postgres -f /docker-entrypoint-initdb.d/03_create_data.sql

  # run the commands file
  docker exec \
    ${CLI_CONTAINER} \
    sh -c "\
      java -jar app.jar < ${TEST_COMMANDS_FILE} > $ACTUAL_RESULTS_FILE && \
      diff -s ${ACTUAL_RESULTS_FILE} ${CORRECT_RESULTS_FILE} > ${DIFFERENCE_FILE}"

  docker cp ${CLI_CONTAINER}:/usr/src/${ACTUAL_RESULTS_FILE} ${OUT_DIR}
  docker cp ${CLI_CONTAINER}:/usr/src/${DIFFERENCE_FILE} ${OUT_DIR}

  FILE_CONTENTS="${OUT_DIR}/${DIFFERENCE_FILE}"
  echo "$(cat ${FILE_CONTENTS})"
else
    echo "File ${TEST_COMMANDS_FILE} does not exist."
fi