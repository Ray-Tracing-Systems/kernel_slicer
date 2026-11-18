//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Graphics pipeline (LiteRS): indirect draws. Counts of these draws read data of the class, so they are computed on GPU right
// before the draw. Every call of a draw command has its own command in 'indirectBuffer' (8 uints), see GPIndirectDraw.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "common{{MainClassSuffixLowerCase}}_slang.h"

RWStructuredBuffer<{{MainClassName}}{{MainClassSuffix}}_UBO_Data> ubo;
RWStructuredBuffer<uint>                                     indirectBuffer;

struct UpdateArgs
{
  uint drawId;
  uint hostCount;    // counts computed on host (arguments and local variables of control function)
  uint hostInstNum;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform UpdateArgs a_args)
{
  uint count   = a_args.hostCount;
  uint instNum = a_args.hostInstNum;
  switch(a_args.drawId)
  {
    {% for Draw in GPIndirectDraws %}
    case {{Draw.Id}}:
      {% if Draw.CountExpr != "" %}
      count   = {{Draw.CountExpr}};
      {% endif %}
      {% if Draw.InstExpr != "" %}
      instNum = {{Draw.InstExpr}};
      {% endif %}
      break;
    {% endfor %}
    default:
      break;
  }

  // VkDrawIndexedIndirectCommand: indexCount, instanceCount, firstIndex, vertexOffset, firstInstance;
  // VkDrawIndirectCommand:        vertexCount, instanceCount, firstVertex, firstInstance
  const uint offset = a_args.drawId*8;
  indirectBuffer[offset + 0] = count;
  indirectBuffer[offset + 1] = instNum;
  indirectBuffer[offset + 2] = 0;
  indirectBuffer[offset + 3] = 0;
  indirectBuffer[offset + 4] = 0;
}
