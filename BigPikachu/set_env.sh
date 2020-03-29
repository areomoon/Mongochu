#!/bin/bash

echo "=== Start downloading images/lables data ==="

python run download_data.py

echo "== Download finished ==="

echo "== Unzip data ==="
unzip Mongochu-master.zip

