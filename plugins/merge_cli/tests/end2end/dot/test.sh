#!/usr/bin/env bash
# Test the partial dot product calculation

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
done

for seed1 in "${seeds[@]}"; do
    for seed2 in "${seeds[@]}"; do
        echo "Computing dot products between seed ${seed1} and seed ${seed2}"
        python dot.py --model-1 "${MODELS[$seed1]}" --model-2 "${MODELS[$seed2]}"
        if [[ ${?} -ne 0 ]]; then
            red_echo "Failure is dot product between seed ${seed1} and seed ${seed2}"
            exit 1
        fi
    done
done

green_echo "dot product test passed."
