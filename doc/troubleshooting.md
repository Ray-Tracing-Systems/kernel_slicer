# Troubleshooting

- **kslicer finished, but the sample does not work.** kslicer returns a non-zero exit code when it
  reports an error, but not every problem is an error. Save the log and check it for `error:` and
  warnings. Then check the output of `build_slang.sh`: a pattern violation often shows up only as a
  shader compile error.
- **`no member named 'X' in namespace 'std'`.** kslicer parses with its own minimal standard
  library in `TINYSTL/`. Use an available function, for example `std::abs` instead of `std::fabs`,
  or add a declaration to `TINYSTL/`: an interface is enough.
- **A function is missing from the generated code.** It does not match a pattern. Check the kernel
  name prefix, the loop shape, and that the control function calls a kernel.
- **kslicer can not find `stddef.h`.** clang can not find its own headers. Copy them next to the
  repository: `mkdir ../lib && cp -r /usr/lib/clang/ ../lib`.
- **`can't open file shaders_generated/....spv` at start.** A shader did not compile. Run
  `build_slang.sh` and read its errors.
- **kslicer fails at step (7) with `basic_ios::clear: iostream error`.** It can not find its
  templates. Run it from the repository root or pass `-selfdir`.
