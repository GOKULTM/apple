#!/bin/bash
pip install --upgrade pip setuptools wheel
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
export PATH="$HOME/.cargo/bin:$PATH"

pip install -r requirements.txt
