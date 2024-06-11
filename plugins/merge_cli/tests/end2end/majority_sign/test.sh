#!/usr/bin/env sh

source ../../../../../tests/end2end/utils.sh

set -e

test_init

echo "Installing git-theta"
git theta install

MODEL_SCRIPT="../model.py"

echo "Making model-1.pt"
python ${MODEL_SCRIPT} --action init --model-name=model-1.pt

echo "Making model-2.pt"
python ${MODEL_SCRIPT} --action init --model-name=model-2.pt

echo "Making model-3.pt"
python ${MODEL_SCRIPT} --action init --model-name=model-3.pt

echo "Inverting model-2.pt"
python invert.py --model model-2.pt

echo "Committing each model"
git theta track model-1.pt
git theta track model-2.pt
git theta track model-3.pt
git add model-1.pt
git add model-2.pt
git add model-3.pt
commit "Commit models to calculate majority sign for"

echo `python majority.py --models model-1.pt model-2.pt model-3.pt`
echo `majority-sign --ties model-1.pt model-2.pt model-3.pt`
