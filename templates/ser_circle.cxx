inline uint get_global_id(uint a_dim) { return glcomp_GlobalInvocationID[a_dim]; }

[[using spirv: buffer, binding(0)]] float out_data[]; 
[[using spirv: buffer, binding(1)]] float in_data[]; 

extern "C" [[using spirv: comp, local_size(256), push]]
void copyKernelFloat(const uint length)
{
  const uint i = get_global_id(0);
  if(i >= length)
    return;
  out_data[i] = in_data[i];
}
