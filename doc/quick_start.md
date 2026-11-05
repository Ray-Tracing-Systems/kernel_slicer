# Quick start

A complete sample is `apps/30_metaballs`, a ray tracer of metaballs with reflections written in the
IPV pattern. Translate it, compile the shaders, then build and run:

```bash
cd apps/30_metaballs
../../cmake-build-release/kslicer $PWD/kmake.json -selfdir $PWD/../..
cd shaders_generated && bash build_slang.sh && cd ..
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release && cmake --build build_release -j 8

./build_release/testapp             # CPU version  -> zout_cpu.bmp
./build_release/testapp --gpu       # GPU version  -> zout_gpu.bmp
./build_release/testapp --compare   # both, timings and image difference
```

kslicer loads its code templates relative to the current directory. Run it from the repository
root, or pass `-selfdir <repository root>` as above.

The same translation without a config file:

```bash
./cmake-build-release/kslicer apps/30_metaballs/metaballs.cpp -mainClass Metaballs \
    -stdlibfolder TINYSTL -Iapps/LiteMath ignore -Iapps/LiteMathAux ignore -ITINYSTL ignore \
    -shaderCC slang -reorderLoops YX -DKERNEL_SLICER -v
```

The generated files are named after the input file: `metaballs_generated.h`,
`metaballs_generated.cpp`, `metaballs_generated_ds.cpp`, `metaballs_generated_init.cpp` and the
`shaders_generated/` folder.

Next: [writing code for kernel_slicer](writing_code.md), [translator options](options.md).
