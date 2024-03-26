TEST_ROOT=$(git rev-parse --show-toplevel)/tests/end2end

TEST_ROOT=${TEST_ROOT} ${TEST_ROOT}/runner.sh
