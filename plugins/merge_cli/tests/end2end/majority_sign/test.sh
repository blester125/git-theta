#!/usr/bin/env bash
# Test the partial sign calculation

source ${TEST_ROOT}/utils.sh

set -e

test_init
echo "Installing git-theta"
git theta install

MODEL_SCRIPT="model.py"

declare -A MODELS
declare -a seeds=(0 42 1337)

for seed in "${seeds[@]}"; do
  echo "Making model with seed ${seed}"
  model=`python ../${MODEL_SCRIPT} --action init --seed ${seed} --model-name=model.pt`
  MODELS[${seed}]="${model}"

  echo "Tracking ${model}"
  git theta track ${model}

  echo "Adding ${model} to git repo."
  git add ${model}
  echo "Committing ${model} to git repo."
  commit "commit ${seed}"

  echo "Computing majority sign for model ${model}"
  python sign.py --model ${model}
  if [[ ${?} -ne 0 ]]; then
    red_echo "Failure in sign calculation in ${model}"
    exit 1
  fi
done

green_echo "Sign calculation test passed!"
