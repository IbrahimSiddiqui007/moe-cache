#!/bin/bash
# Builds gpu_keeper for Linux and Windows into build/. Needs glslc; the Windows one needs mingw-w64.
set -e
cd "$(dirname "$0")"; mkdir -p build
glslc -O --target-env=vulkan1.0 -mfmt=c work.comp -o build/work.inc
{ echo "static const uint32_t work_spv[] ="; cat build/work.inc; echo ";"; } > build/work_spv.h
cc -O2 -Wall -Ibuild gpu_keeper.c -o build/gpu_keeper -ldl
mkdir -p build/inc; ln -sfn /usr/include/vulkan build/inc/vulkan; ln -sfn /usr/include/vk_video build/inc/vk_video
x86_64-w64-mingw32-gcc -O2 -Wall -static -Ibuild -Ibuild/inc gpu_keeper.c -o build/gpu_keeper.exe
x86_64-w64-mingw32-gcc -O2 -Wall -static timer_res.c -o build/timer_res.exe
echo built
