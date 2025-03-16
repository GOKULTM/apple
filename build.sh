#!/bin/bash
pip install --upgrade pip setuptools wheel
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
pip install -r requirements.txt
