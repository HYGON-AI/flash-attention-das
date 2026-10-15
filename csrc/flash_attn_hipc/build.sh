#!/bin/bash

# Protect codes
if [ -d "./src" ] && [ -d "./include" ]; then
    RED='\033[31m'
    NC='\033[0m'
    echo -e "${RED}\nPlease mkdir build && cd build for building flash-attention!\n${NC}"
    exit
fi

# Clean ever generated files
rm -rf ./CMakeFiles
rm -rf ./.ninja*
rm -rf ./*.ninja
rm -rf ./cmake_*
rm -rf ./CMakeCache.txt
rm -rf ./*.*o
rm -rf ./*.asm

# Acquire device id or arch id to decide compiler behaviors
target=$(rocminfo | grep gfx)
echo $target
if [[ $target =~ gfx([0-9a-zA-Z]+) ]]; then
    gfx_seq=${BASH_REMATCH[1]}
fi
if [[ "$gfx_seq" == "9210" ]]; then
    gfx_seq="92a"
fi
if [ -z "$gfx_seq" ]; then
    target=$(lspci | grep 'Co-processor:')
    echo $target
    start_pos=${target#*Device }
    device_id=${start_pos%% *}
    if [[ "$device_id" == "6320" || "$device_id" == "6321" || "$device_id" == *"BW"* ]]; then
        gfx_seq="936"
    elif [[ "$device_id" == "6210" || "$device_id" == "6211" ]]; then
        gfx_seq="928"
    else
        gfx_seq="928;936;938"
    fi
fi
build_c_string="BUILD_C_INTERFACE=ON"
if [[ "$2" =~ $build_c_string ]]; then
    gfx_seq="928;936;938;92a"
fi

echo gfx$gfx_seq

# Decide build options
build_options=""
if [[ "$1" == "1" ]]; then
    build_options="-DBUILD_FA_FWD=ON "
elif [[ "$1" == "2" ]]; then
    build_options="-DBUILD_FA_BWD=ON "
elif [[ "$1" == "3" ]]; then
    build_options="-DBUILD_FA_KVCACHE=ON "
elif [[ "$1" == "4" ]]; then
    build_options="-DBUILD_FA_PERMUTE=ON "
elif [[ "$1" == "5" ]]; then
    build_options="-DBUILD_FLASHMLA=ON "
else
    build_options="-DBUILD_FA_FWD=ON -DBUILD_FA_BWD=ON -DBUILD_FA_KVCACHE=ON -DBUILD_FA_PERMUTE=ON -DBUILD_FLASHMLA=ON "
fi

# Generate makefiles and build
extra_options=${@:2}
echo "extra_options: $extra_options"
if [[ "$extra_options" == *"ninja"* ]]; then
    extra_options=${extra_options//ninja/}
    cmake .. -DBUILD_ASM=ON -G Ninja -DFA_DEBUG=ON -DGFX_VERSION=$gfx_seq $build_options $extra_options
    ninja -j8 -v
else
    cmake .. -DBUILD_ASM=ON -DFA_DEBUG=ON -DGFX_VERSION=$gfx_seq $build_options $extra_options
    VERBOSE=1 make -j8
fi
echo "OK"