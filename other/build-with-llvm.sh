#!/usr/bin/bash

# Exit if llvm-project is already exists
if [[ -d llvm-project ]]; then
    echo "llvm-project directory exists, exiting installation process"
    exit 1
fi

# Download LLVM-17 and extract
echo "Downloading LLVM-17 source code"
if ! wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-17.0.6.tar.gz; then
    echo "Failed to download LLVM-17"
    exit 1
fi
tar -xzvf llvmorg-17.0.6.tar.gz
rm llvmorg-17.0.6.tar.gz
mv llvm-project-llvmorg-17.0.6 llvm-project

# Go to llvm-project
cd llvm-project

# Add kernel_slicer to LLVM
echo "Cloning kernel_slicer source code"
if ! git clone https://github.com/Ray-Tracing-Systems/kernel_slicer.git ./clang-tools-extra/kernel_slicer --recurse-submodules; then
    echo "Failed to clone \`kernel_slicer\`, exiting"
    exit 1
fi
echo "Adding \`kernel_slicer\` subdirectory to clang-tools-extra's CMakeLists.txt"
printf "\nadd_subdirectory(kernel_slicer)\n" >> ./clang-tools-extra/CMakeLists.txt
