#!/usr/bin/env bash


GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NORMAL="\033[0m"

function color_echo {
    local text="${1}"
    local color="${2}"
    echo -e "${color}${text}${NORMAL}"
}

function green_echo {
    local text="${1}"
    color_echo "${text}" "${GREEN}"
}

function red_echo {
    local text="${1}"
    color_echo "${text}" "${RED}"
}

function yellow_echo {
    local text="${1}"
    color_echo "${text}" "${YELLOW}"
}

function make_repo {
    echo "Making Git Repo."
    git init
    git branch -m main
}

function test_init {
    make_repo
    if [[ ! ${-} =~ e ]]; then
        red_echo "It seems that 'set -e' was not done in this test. This makes it easy for a test to fail but appear to pass. Please add 'set -e' and do specific error handling if a part of your test is allowed to fail."
        exit 1
    fi
}

function commit {
    local msg="${1}"
    git commit -m "${msg}" > /dev/null
    echo $(git rev-parse HEAD)
}