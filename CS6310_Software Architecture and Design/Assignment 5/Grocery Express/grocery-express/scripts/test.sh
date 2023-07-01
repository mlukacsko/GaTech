#!/usr/bin/env bash

DOCKER_COMPOSE_PREFIX="grocery-express"
DOCKER_COMPOSE_POSTGRES_HOST="db"

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
    "${DOCKER_COMPOSE_PREFIX}-db-1" \
    psql -U postgres -f /docker-entrypoint-initdb.d/01_truncate_tables.sql

  # repopulate initial data
  docker exec \
    "${DOCKER_COMPOSE_PREFIX}-db-1" \
    psql -U postgres -f /docker-entrypoint-initdb.d/03_create_data.sql

  # run the commands file
  docker run \
      -e "POSTGRES_HOST=${DOCKER_COMPOSE_POSTGRES_HOST}" \
      --env-file .env \
      --network "${DOCKER_COMPOSE_PREFIX}_default" \
      --name groceryexpress-cli-tester \
      "${DOCKER_COMPOSE_PREFIX}-cli" \
      sh -c "\
        java -jar app.jar < ${TEST_COMMANDS_FILE} > ${ACTUAL_RESULTS_FILE} && \
        diff -s ${ACTUAL_RESULTS_FILE} ${CORRECT_RESULTS_FILE} > ${DIFFERENCE_FILE}"

  docker cp groceryexpress-cli-tester:/usr/src/${ACTUAL_RESULTS_FILE} ${OUT_DIR}
  docker cp groceryexpress-cli-tester:/usr/src/${DIFFERENCE_FILE} ${OUT_DIR}
  docker rm groceryexpress-cli-tester > /dev/null

  FILE_CONTENTS="${OUT_DIR}/${DIFFERENCE_FILE}"
  echo "$(cat ${FILE_CONTENTS})"
else
    echo "File ${TEST_COMMANDS_FILE} does not exist."
fi