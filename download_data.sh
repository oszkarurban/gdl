#!/bin/bash

mkdir -p data/cath
# mkdir -p data/ts
mkdir -p results/ProDesign

# Download CATH 4.2
echo "Downloading CATH 4.2 dataset"
wget -O data/cath.zip https://github.com/A4Bio/PiFold/releases/download/Training%26Data/cath4.2.zip
echo "Unzipping CATH"
unzip -q data/cath.zip -d data/cath
# Move nested files up if necessary (Colab script implies a nested cath4.2 folder)
if [ -d "data/cath/cath4.2" ]; then
    mv data/cath/cath4.2/* data/cath/
    rmdir data/cath/cath4.2
fi
rm data/cath.zip

# # Download TS dataset
# echo "Downloading TS dataset"
# wget -O data/ts.zip https://github.com/A4Bio/PiFold/releases/download/Training%26Data/ts.zip
# echo "Unzipping TS"
# unzip -q data/ts.zip -d data/
# rm data/ts.zip

echo "Downloading checkpoint"
wget -O results/ProDesign/checkpoint.pth https://github.com/A4Bio/PiFold/releases/download/Training%26Data/checkpoint.pth

echo "complete"
