#!/bin/bash
set -e

echo "===== Step1 Update system package source ====="
sudo apt update -y

echo "===== Step2 Install Python3, pip & git ====="
sudo apt install -y python3 python3-pip git

echo "===== Step3 Clone AICP CLI repository ====="
git clone https://github.com/woozheng/aicp_cli.git
cd aicp_cli

echo "===== Step4 Install all dependencies ====="
pip3 install -r requirements.txt

echo "===== Step5 Edit config file (set api_key & model) ====="
echo "Opening aicp.yaml with nano, fill your API key and model name manually"
nano aicp.yaml

echo "===== Step6 Launch AICP CLI ====="
python3 aicp.py