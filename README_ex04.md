
# Example #4: virtual kernell call (NOT YET IMPLEMENTED!)

In the previous example we have considered special case for changing control flow by breaking the loop. But what if you need a bit different control flow change: 

```cpp
void TestClass::PathTrace(uint tid, float4* out_color)
{
  if(kernel_XXX(tid, ...))
    kernel_YYY(tid, out_color);
  else
    kernel_ZZZ(tid, out_color);
  ...
}
```

In fact this is quite standart case for many applications which has some general form. Therefore we have implemented a more general way for such code which we have called **virtual kernel calls**. Instead of putting conditions and branches you should use object oriented approach:

```cpp
void TestClass::PathTrace(uint tid, float4* out_color)
{
  IMaterial* pMaterial = kernel_XXX(tid, ...);

  pMaterial->kernel_YYY_or_ZZZ();
  ...
}
```

It is unnecessary to recall that there can be as many implementadtion classes as you like, and that "kernel_YYY_or_ZZZ" assumed to be virtual function now. There are at least three ways for implementing such virtual kernel calls effitiently on GPU via Vulkan:

1. Just transform virtual function call to switch statement inside kernel. Quite obvious but still can be effitient enough for lite functions;

2. Sort threads and call correct kernels via IndirectDispatch;

3. Use RTX API in Vulkan and put all virtual function implementations in different surface shaders.

All these 3 methods have their advantages and disadvantages and we plan to implement all off them. As user you can select on of these ways via translator presets. It should be noted that you don't have to change your source code.

