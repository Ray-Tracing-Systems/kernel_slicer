# Common mistakes and their fixes

Each item shows code that breaks translation (or produces wrong GPU results) and the fixed form.

## 1. Two consecutive loops in one kernel

This was produced by an earlier AI attempt (`apps/ai_generated/01_image_gauss_blur`).

```cpp
// WRONG: two top-level loops, plus a sum that depends on all previous iterations
void GaussianBlur::kernel1D_generateKernel(int kernelRadius, float sigma)
{
  float sum = 0.0f;
  for(int i = -kernelRadius; i <= kernelRadius; i++) { m_kernel[i + kernelRadius] = exp(...); sum += m_kernel[i + kernelRadius]; }
  for(int i = 0; i < 2*kernelRadius + 1; i++)        { m_kernel[i] /= sum; }
}
```

Fixes, in order of preference:
- Filter weights depend only on parameters: compute them on the host (constructor or a setter)
  into a member vector, then call `CommitDeviceData()`. No kernel is needed.
- If they must be computed on the GPU, split into kernels and use a member reduction:

```cpp
void GaussianBlur::kernel1D_Weights(int a_size, int a_radius, float a_sigma)
{
  m_weightSum = 0.0f;
  for(int i = 0; i < a_size; i++)
  {
    const float d = float(i - a_radius);
    m_weights[i]  = exp(-d*d / (2.0f*a_sigma*a_sigma));
    m_weightSum  += m_weights[i];              // sum-reduction into a member
  }
}
void GaussianBlur::kernel1D_Normalize(int a_size)
{
  for(int i = 0; i < a_size; i++)
    m_weights[i] = m_weights[i] / m_weightSum;
}
```

## 2. Loop bound is an expression

Verified: kslicer accepts this without any message, but the generated shaders fail in
the shader build script with a syntax error at the `*`.

```cpp
// WRONG
for(int i = 0; i < w*h; i++) { ... }
// RIGHT: pass the product as an argument from the control function
void MyClass::kernel1D_Process(int a_size, ...) { for(int i = 0; i < a_size; i++) { ... } }
...
kernel1D_Process(w*h, ...);
```

## 3. Temporary vector inside a control function

```cpp
// WRONG
void MyClass::Run(int n, const float* in [[size("n")]], float* out [[size("n")]])
{
  std::vector<float> tmp(n);
  kernel1D_A(n, in, tmp.data());
  kernel1D_B(n, tmp.data(), out);
}
// RIGHT: member vector, sized before CommitDeviceData()
std::vector<float> m_tmp;                         // in the class
void MyClass::Reserve(int n) { m_tmp.resize(n); } // host-only setter
void MyClass::Run(int n, const float* in [[size("n")]], float* out [[size("n")]])
{
  kernel1D_A(n, in, m_tmp.data());
  kernel1D_B(n, m_tmp.data(), out);
}
```

## 4. In-out pointer argument

```cpp
// WRONG: 'data' is read and written, but a non-const pointer is treated as output only
virtual void Scale(float* data [[size("n")]], int n);
// RIGHT
virtual void Scale(const float* a_in [[size("n")]], float* a_out [[size("n")]], int n);
```

If an algorithm needs in-place updates across passes, keep the working copy in a member vector,
copy the input in with a kernel or `memcpy`, and copy the result out at the end.

## 5. Missing size annotation or missing `const`

```cpp
// WRONG
virtual void Run(int w, int h, float4* in, uint32_t* out);
// RIGHT
virtual void Run(int w, int h, const float4* in [[size("w*h")]], uint32_t* out [[size("w*h")]]);
```

## 6. Kernel calls a kernel / control calls a control

```cpp
// WRONG
void C::kernel1D_A(int n, float* p) { for(int i = 0; i < n; i++) { ... } kernel1D_B(n, p); }
// RIGHT: sequence them in the control function
void C::Run(...) { kernel1D_A(n, p); kernel1D_B(n, p); }
```

Shared logic goes into a normal helper function that both may call.

## 7. Non-perfect loop nest in a 2D kernel

```cpp
// WRONG: statement between the two thread loops
for(int y = 0; y < h; y++) { float rowScale = m_rows[y]; for(int x = 0; x < w; x++) { ... } }
// RIGHT: move it into the inner body
for(int y = 0; y < h; y++) for(int x = 0; x < w; x++) { float rowScale = m_rows[y]; ... }
```

## 8. Cross-thread dependency inside one kernel

A thread must not read data written by *another* thread of the same kernel launch (neighbors of
a stencil output, a running prefix sum, the result of a previous row). GPU threads run in no
particular order. Split into kernels (every kernel boundary is a global barrier), use a member
reduction, `InterlockedAdd`, or the `std::exclusive_scan` service call.

## 9. Pointer members and nested containers

```cpp
// WRONG: ignored by the translator / not supported
float* m_weights;
struct Mesh { std::vector<float3> verts; };  std::vector<Mesh> m_meshes;
// RIGHT: flat member vectors plus offset/count tables
std::vector<float4>   m_allVerts;
std::vector<uint32_t> m_meshOffset, m_meshCount;
```

## 10. float3 in buffers and structs

```cpp
// RISKY: std430 alignment of vec3 differs from C++ (12 vs 16 bytes)
struct Particle { float3 pos; float3 vel; float mass; };
// SAFE
struct Particle { float4 posAndMass; float4 vel; };
```

## 11. Local vector / heap / I/O in GPU code

```cpp
// WRONG inside kernels and helpers
std::vector<float> neighbors;  std::cout << x;  throw ...;  new Foo;
// RIGHT
float neighbors[8];            // fixed size; [[threadlocal]] for larger arrays
```

## 12. Wrong call order on the host

```cpp
// WRONG: vectors resized after upload; the GPU buffers keep the old (empty) size
pImpl->CommitDeviceData();
pImpl->SetMaxImageSize(w, h);
// RIGHT
pImpl->SetMaxImageSize(w, h);
pImpl->CommitDeviceData();
```

Every later change of member data on the host needs another `CommitDeviceData()`.

## 13. Using RTV names by accident

A function named `kernel_Something` is an RTV kernel and must take `tid`. For IPV always use
`kernel1D_`/`kernel2D_`/`kernel3D_`. Also avoid the substring `kernel` followed by those
prefixes in helper function names: any name containing `kernel_`, `kernel1D_`... is treated as a
kernel.

## 14. Double precision and literals

`1.0` is a `double` literal and may force double arithmetic in shaders. Write `1.0f` and use
`float` unless the task explicitly requires double and the GPU supports it.

## 15. `std::` function missing from TINYSTL

```cpp
// WRONG: kslicer's parser stops with "no member named 'fabs' in namespace 'std'"
float a = std::fabs(c.x);
// RIGHT
float a = std::abs(c.x);
```

kslicer still exits with code 0 and writes output, so check its log for `error:`.

## 16. 2D work-group size as the default for all kernels

```jsonc
// WRONG: also applied to kernel1D_ reductions, which then give wrong results on GPU
"kernels" : { "default" : {"wgSize": [32, 8, 1]} }
// RIGHT: keep the defaults, override only named 2D kernels
"kernels" : { "kernel2D_Convolve" : {"wgSize": [16, 16, 1]} }
```

## 17. Member or kernel argument inside a `std::` call (Slang back end)

Verified with `-shaderCC slang` (GLSL is not affected):

```cpp
// WRONG: generated Slang is  max(ubo[0, x)  and  pow(x, kge)  -> slangc error 30027
a_out[i] = std::max(m_val, x);
a_out[i] = std::pow(x, a_p);
// RIGHT: copy to locals, then call
const float val = m_val;
const float p   = a_p;
a_out[i] = std::max(val, x);
a_out[i] = std::pow(x, p);
```

The same call without `std::` (`pow(x, a_p)`) is translated correctly, but on the CPU side
unqualified `abs`/`max` may pick the wrong overload, so local copies are the safer fix.
The reduction statement `m_max = std::max(m_max, v);` is handled by a separate path and works.
