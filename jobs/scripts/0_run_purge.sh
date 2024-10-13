#!/bin/bash
echo "Removing all results and recreating folders..."
rm -rf ../../data/out/
rm -rf ../../data/models/
rm -f ../../log.sh
mkdir ../../data/out/ ../../data/models/
touch ../../log.sh