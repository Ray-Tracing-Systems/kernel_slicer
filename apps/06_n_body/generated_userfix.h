#ifndef TEST_GENERATED_USERFIX_H
#define TEST_GENERATED_USERFIX_H

#include "test_class_generated.h"

class nBody_GeneratedFix : public nBody_Generated
{
public:
  void performCmd(VkCommandBuffer a_commandBuffer, BodyState *out_bodies) override;
  void performCmdCustom(bool &init, VkCommandBuffer a_commandBuffer, BodyState *out_bodies);
};

#endif //TEST_GENERATED_USERFIX_H
