#include "codegen_host.h"
// #include "class.h"
#include "class_info.h"
#include "codegen.h"
#include "utils.h"
#include <clang/AST/AST.h>
#include <inja.hpp>

namespace common_rewriter {
    using namespace nlohmann;
    using llvm::outs;

    void codegen(const ClassInfo& info) {

        json data = json::object();

        /* -------------------------------------------------------------------------- */
        /*                                   Kernels                                  */
        /* -------------------------------------------------------------------------- */

        data["Kernels"] = json::array();
        for (const KernelInfo& i : info.kernels) {
            auto kernel = json::object();
            kernel["Name"] = i.decl->getNameAsString().substr(9);
            kernel["OriginalName"] = i.decl->getNameAsString();
            kernel["IsMega"] = false;
            kernel["UseRayGen"] = false;
            kernel["FinishRed"] = false;
            kernel["HasLoopInit"] = false;
            kernel["HasLoopFinish"] = false;
            kernel["IsIndirect"] = false;

            {
                std::string decl;
                decl += i.decl->getNameAsString().substr(9) + "Cmd";
                decl += "(";
                for (size_t j = 0; j < i.decl->getNumParams(); j++) {
                    decl += render_type_name(i.decl->getParamDecl(j)->getType());
                    decl += " ";
                    decl += i.decl->getParamDecl(j)->getNameAsString();
                    if (j < i.decl->getNumParams() - 1)
                        decl += ", ";
                }
                decl += ")";
                kernel["Decl"] = decl;
            }

            if (i.decl->getNameAsString().substr(6, 1) == "1") {
                kernel["WGSizeX"] = 256;
                kernel["WGSizeY"] = 1;
                kernel["WGSizeZ"] = 1;
            } else if (i.decl->getNameAsString().substr(6, 1) == "2") {
                kernel["WGSizeX"] = 32;
                kernel["WGSizeY"] = 8;
                kernel["WGSizeZ"] = 1;
            } else if (i.decl->getNameAsString().substr(6, 1) == "3") {
                kernel["WGSizeX"] = 8;
                kernel["WGSizeY"] = 8;
                kernel["WGSizeZ"] = 8;
            } else {
                REWRITE_ASSERT(false);
            }

            kernel["AuxArgs"] = json::array();
            for (size_t j = 0; j < i.decl->getNumParams(); j++) {
                auto p = i.decl->getParamDecl(j);
                if (std::find(i.bounds_args.begin(), i.bounds_args.end(), p) == i.bounds_args.end() &&
                    !p->getType()->isPointerType()) {
                    auto arg = json::object();
                    arg["Name"] = p->getNameAsString();
                    arg["Type"] = render_type_name(p->getType());
                    kernel["AuxArgs"].push_back(arg);
                }
            }

            kernel["SmplX"] = true;
            kernel["SmplY"] = true;
            kernel["SmplZ"] = true;
            kernel["tidX"] = i.bounds_args.size() > 0 ? i.bounds_args[0]->getNameAsString() : "1";
            kernel["tidY"] = i.bounds_args.size() > 1 ? i.bounds_args[1]->getNameAsString() : "1";
            kernel["tidZ"] = i.bounds_args.size() > 2 ? i.bounds_args[2]->getNameAsString() : "1";
            kernel["EnableBlockExpansion"] = true;

            auto args = json::array();
            for (auto binding : i.bindings) {
                auto arg = json::object();
                arg["Name"] = binding->getNameAsString();
                arg["Id"] = args.size();
                arg["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
                arg["IsTextureArray"] = false;
                arg["Count"] = 1;
                args.push_back(arg);
            }
            kernel["Args"] = args;
            kernel["ArgCount"] = args.size();
            kernel["IsRTV"] = false;

            data["Kernels"].push_back(kernel);
        }

        data["KernelsDecls"] = json::array();
        for (const KernelInfo& i : info.kernels) {
            std::string decl;
            decl += "virtual ";
            decl += render_type_name(i.decl->getReturnType());
            decl += " ";
            decl += i.decl->getNameAsString().substr(9) + "Cmd";
            decl += "(";
            for (size_t j = 0; j < i.decl->getNumParams(); j++) {
                decl += render_type_name(i.decl->getParamDecl(j)->getType());
                decl += " ";
                decl += i.decl->getParamDecl(j)->getNameAsString();
                if (j < i.decl->getNumParams() - 1)
                    decl += ", ";
            }
            decl += ");";
            data["KernelsDecls"].push_back(decl);
        }

        /* -------------------------------------------------------------------------- */
        /*                              Control functions                             */
        /* -------------------------------------------------------------------------- */

        data["DescriptorSetsAll"] = json::array();
        // for (auto [control, kernels] : controls_kernel_calls) {
        //     for (auto kernel : kernels) {
        //         auto ds = json::object();
        //         ds["Layout"] = kernel->getNameAsString();
        //         ds["Id"] = ds_count++;
        //         data["DescriptorSetsAll"].push_back(ds);
        //     }
        // }

        size_t descriptors_count = 0;
        size_t descriptor_sets_count = 0;

        // size_t ds_id = 0;
        data["MainFunctions"] = json::array();
        for (auto i : info.control_functions) {
            auto out = json::object();
            out["IsRTV"] = false;
            out["IsMega"] = false;
            out["LocalVarsBuffersDecl"] = {};
            out["Name"] = i.decl->getNameAsString();
            out["ReturnType"] = render_type_name(i.decl->getReturnType());

            std::string decl;

            auto in_outs = json::array();
            for (size_t j = 0; j < i.decl->getNumParams(); j++) {
                auto v = i.decl->getParamDecl(j);
                decl += render_type_name(v->getType());
                decl += " ";
                decl += v->getNameAsString();
                if (j < i.decl->getNumParams() - 1)
                    decl += ", ";
                if (v->getType()->isPointerType()) {
                    auto var = json::object();
                    var["IsTexture"] = false;
                    var["Name"] = v->getNameAsString();
                    in_outs.push_back(var);
                }
            }
            out["DeclOrig"] = i.decl->getNameAsString() + "(" + decl + ")";
            out["Decl"] = i.decl->getNameAsString() + "Cmd(VkCommandBuffer a_commandBuffer, " + decl + ")";
            out["MainFuncDeclCmd"] = out["Decl"];
            out["InOutVars"] = in_outs;
            out["OverrideMe"] = true;

            ControlFunctionCodegen codegen(info, i.decl, descriptor_sets_count);
            codegen.render_statement(i.decl->getBody(), 0);
            out["MainFuncTextCmd"] = codegen.buffer;

            out["FullImpl"] = json::object();
            out["FullImpl"]["InputData"] = json::array();
            out["FullImpl"]["OutputData"] = json::array();
            std::string in_out = "";
            for (size_t j = 0; j < i.decl->getNumParams(); j++) {
                auto p = i.decl->getParamDecl(j);
                if (p->getType()->isPointerType()) {
                    auto var = json::object();
                    var["IsTexture"] = false;
                    var["Name"] = p->getNameAsString();
                    var["DataSize"] = "width * height";
                    var["DataType"] = render_type_name(p->getType()->getPointeeType());
                    out["FullImpl"]["OutputData"].push_back(var);
                    in_out += p->getNameAsString() + "GPU" + ", 0, ";
                }
            }
            out["FullImpl"]["ArgsOnSetInOut"] = in_out + "0";
            out["FullImpl"]["HasImages"] = false;
            {
                std::string args;
                for (size_t j = 0; j < i.decl->getNumParams(); j++) {
                    args += i.decl->getParamDecl(j)->getNameAsString();
                    if (j < i.decl->getNumParams() - 1)
                        args += ", ";
                }
                out["FullImpl"]["ArgsOnCall"] = args;
            }

            out["DescriptorSets"] = json::array();

            for (auto call_expr : i.kernel_calls) {
                const KernelInfo* kernel = info.get_kernel(call_expr->getMethodDecl());
                auto ds = json::object();
                ds["Id"] = descriptor_sets_count++;
                ds["KernelName"] = kernel->decl->getNameAsString();
                ds["ArgNames"] = "ARG_NAMES()";
                ds["IsServiceCall"] = false;
                ds["IsVirtual"] = false;
                ds["ArgNumber"] = kernel->bindings.size();
                ds["Layout"] = kernel->decl->getNameAsString();
                ds["Args"] = json::array();

                size_t index = 0;
                for (auto binding : kernel->bindings) {
                    auto arg = json::object();
                    arg["IsTexture"] = false;
                    arg["IsTextureArray"] = false;
                    arg["IsAccelStruct"] = false;
                    arg["Id"] = index++;
                    arg["Name"] = binding->getNameAsString();
                    descriptors_count++;
                    ds["Args"].push_back(arg);
                }
                // outs() << ds["Args"].size() << "\n";
                data["DescriptorSetsAll"].push_back(ds);
                out["DescriptorSets"].push_back(ds);
            }

            data["MainFunctions"].push_back(out);
        }
        data["TotalBuffersUsed"] = descriptors_count;
        data["TotalDSNumber"] = descriptor_sets_count;

        /* -------------------------------------------------------------------------- */
        /*                               Uniform fields                               */
        /* -------------------------------------------------------------------------- */

        data["UBO"] = json::object();
        data["UBO"]["UBOStructFields"] = json::array();
        for (auto i : info.used_uniform_fields) {
            auto field = json::object();
            field["Name"] = i->getNameAsString();
            field["Type"] = render_type_name(i->getType());
            field["IsArray"] = get_type_array_size(i->getType()) > 0;
            field["ArraySize"] = get_type_array_size(i->getType());
            field["IsDummy"] = false;
            data["UBO"]["UBOStructFields"].push_back(field);
        }

        data["ClassVars"] = json::array();
        for (auto i : info.used_uniform_fields) {
            auto var = json::object();
            var["IsArray"] = i->getType()->isArrayType();
            var["HasPrefix"] = false;
            var["IsConst"] = false;
            var["Name"] = i->getNameAsString();
            data["ClassVars"].push_back(var);
        }

        /* -------------------------------------------------------------------------- */
        /*                                Vector fields                               */
        /* -------------------------------------------------------------------------- */

        data["ClassVectorVars"] = json::array();
        for (auto i : info.used_buffer_fields) {
            auto buffer = json::object();
            buffer["Name"] = i->getNameAsString();
            buffer["IsVFHBuffer"] = false;
            buffer["VFHLevel"] = 0;
            buffer["AccessSymb"] = ".";
            buffer["WithBuffRef"] = false;
            buffer["TypeOfData"] = render_type_name(get_vector_specialization_type(i->getType()));
            data["ClassVectorVars"].push_back(buffer);
        }

        data["VectorMembers"] = json::array();
        for (auto i : info.used_buffer_fields) {
            auto buffer = json::object();
            buffer["Name"] = i->getNameAsString();
            buffer["IsVFHBuffer"] = false;
            buffer["VFHLevel"] = 0;
            buffer["HasPrefix"] = false;
            data["VectorMembers"].push_back(buffer);
        }

        data["UseServiceScan"] = false;
        data["UseServiceSort"] = false;
        data["HasAllRefs"] = false;
        data["UseRayGen"] = false;
        data["IncludeClassDecl"] = "D:/kernel_slicer/class.h";
        data["SceneMembers"] = json::array();
        data["ForceProxy"] = false;
        data["IntersectionHierarhcy"] = json::object();
        data["IntersectionHierarhcy"]["Implementations"] = json::array();
        data["HasIntersectionShaders"] = false;
        data["Constructors"] = json::array();
        {
            auto ctor = json::object();
            ctor["NumParams"] = 0;
            ctor["ClassName"] = "VolumeRenderer";
            data["Constructors"].push_back(ctor);
        }

        data["MainClassName"] = "VolumeRenderer";
        data["MainClassSuffix"] = "_slang";

        data["SlangShaders"] = true;
        data["UseVulkan11"] = false;

        data["SpecConstants"] = json::array();

        data["ShaderSingleFile"] = "";

        data["EnableTimeStamps"] = false;

        data["HaveLocalContainers"] = false;

        data["UseSubGroups"] = false;

        data["HasFullImpl"] = true;

        data["RemapTables"] = json::array();

        data["UseSeparateUBO"] = false;

        // data["ClassVectorVars"] = json::array();
        // for (auto i : buffer_fields) {
        //     auto buffer = json::object();
        //     buffer["Name"] = i->getNameAsString();
        //     buffer["IsVFHBuffer"] = false;
        //     buffer["VFHLevel"] = 0;
        //     buffer["AccessSymb"] = ".";
        //     buffer["WithBuffRef"] = false;
        //     buffer["TypeOfData"] = render_type_name(get_vector_specialization_type(i->getType()));
        //     data["ClassVectorVars"].push_back(buffer);
        // }

        // data["VectorMembers"] = json::array();
        // for (auto i : buffer_fields) {
        //     auto buffer = json::object();
        //     buffer["Name"] = i->getNameAsString();
        //     buffer["IsVFHBuffer"] = false;
        //     buffer["VFHLevel"] = 0;
        //     buffer["HasPrefix"] = false;
        //     data["VectorMembers"].push_back(buffer);
        // }

        data["TexArrayMembers"] = json::array();

        data["ClassTextureVars"] = json::array();
        data["ClassTexArrayVars"] = json::array();
        data["SamplerMembers"] = json::array();
        data["RedVectorVars"] = json::array(); // ???
        data["IndirectDispatches"] = json::array();

        data["MultipleSourceShaders"] = true;
        data["ShaderFolder"] = "./shaders_slang/";

        // data["Kernels"] = json::array();
        // for (auto i : kernels) {
        //     auto kernel = json::object();
        //     kernel["Name"] = i->getNameAsString().substr(9);
        //     kernel["OriginalName"] = i->getNameAsString();
        //     kernel["IsMega"] = false;
        //     kernel["UseRayGen"] = false;
        //     kernel["FinishRed"] = false;
        //     kernel["HasLoopInit"] = false;
        //     kernel["HasLoopFinish"] = false;
        //     kernel["IsIndirect"] = false;

        //     {
        //         std::string decl;
        //         // decl += "virtual ";
        //         // decl += render_type_name(i->getReturnType());
        //         // decl += " ";
        //         decl += i->getNameAsString().substr(9) + "Cmd";
        //         decl += "(";
        //         for (size_t j = 0; j < i->getNumParams(); j++) {
        //             decl += render_type_name(i->getParamDecl(j)->getType());
        //             decl += " ";
        //             decl += i->getParamDecl(j)->getNameAsString();
        //             if (j < i->getNumParams() - 1)
        //                 decl += ", ";
        //         }
        //         decl += ")";
        //         kernel["Decl"] = decl;
        //     }

        //     if (i->getNameAsString().substr(6, 1) == "1") {
        //         kernel["WGSizeX"] = 256;
        //         kernel["WGSizeY"] = 1;
        //         kernel["WGSizeZ"] = 1;
        //     } else if (i->getNameAsString().substr(6, 1) == "2") {
        //         kernel["WGSizeX"] = 32;
        //         kernel["WGSizeY"] = 8;
        //         kernel["WGSizeZ"] = 1;
        //     } else if (i->getNameAsString().substr(6, 1) == "3") {
        //         kernel["WGSizeX"] = 8;
        //         kernel["WGSizeY"] = 8;
        //         kernel["WGSizeZ"] = 8;
        //     } else {
        //         REWRITE_ASSERT(false);
        //     }

        //     kernel["AuxArgs"] = json::array();
        //     for (size_t j = 0; j < i->getNumParams(); j++) {
        //         auto p = i->getParamDecl(j);
        //         if (!contains(kernels_dimentions[i], p) && !p->getType()->isPointerType()) {
        //             auto arg = json::object();
        //             arg["Name"] = p->getNameAsString();
        //             arg["Type"] = render_type_name(p->getType());
        //             kernel["AuxArgs"].push_back(arg);
        //         }
        //     }

        //     kernel["SmplX"] = true;
        //     kernel["SmplY"] = true;
        //     kernel["SmplZ"] = true;
        //     kernel["tidX"] = kernels_dimentions[i].size() > 0 ? kernels_dimentions[i][0]->getNameAsString() : "1";
        //     kernel["tidY"] = kernels_dimentions[i].size() > 1 ? kernels_dimentions[i][1]->getNameAsString() : "1";
        //     kernel["tidZ"] = kernels_dimentions[i].size() > 2 ? kernels_dimentions[i][2]->getNameAsString() : "1";
        //     kernel["EnableBlockExpansion"] = true;

        //     auto args = json::array();
        //     for (auto binding : kernels_bindings[i]) {
        //         auto arg = json::object();
        //         arg["Name"] = binding->getNameAsString();
        //         arg["Id"] = args.size();
        //         arg["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
        //         arg["IsTextureArray"] = false;
        //         arg["Count"] = 1;
        //         args.push_back(arg);
        //     }
        //     kernel["Args"] = args;
        //     kernel["ArgCount"] = args.size();
        //     kernel["IsRTV"] = false;

        //     data["Kernels"].push_back(kernel);
        // }

        // data["kernelsDecls"] = json::array();
        // for (auto i : kernels) {
        //     std::string decl;
        //     decl += "virtual ";
        //     decl += render_type_name(i->getReturnType());
        //     decl += " ";
        //     decl += i->getNameAsString().substr(9) + "Cmd";
        //     decl += "(";
        //     for (size_t j = 0; j < i->getNumParams(); j++) {
        //         decl += render_type_name(i->getParamDecl(j)->getType());
        //         decl += " ";
        //         decl += i->getParamDecl(j)->getNameAsString();
        //         if (j < i->getNumParams() - 1)
        //             decl += ", ";
        //     }
        //     decl += ");";
        //     data["KernelsDecls"].push_back(decl);
        // }

        // size_t ds_count = 0;
        // data["DescriptorSetsAll"] = json::array();
        // for (auto [control, kernels] : controls_kernel_calls) {
        //     for (auto kernel : kernels) {
        //         auto ds = json::object();
        //         ds["Layout"] = kernel->getNameAsString();
        //         ds["Id"] = ds_count++;
        //         data["DescriptorSetsAll"].push_back(ds);
        //     }
        // }
        // data["TotalDSNumber"] = ds_count;
        // size_t desc_count = 0;

        // size_t ds_id = 0;
        // data["MainFunctions"] = json::array();
        // for (auto i : controls) {
        //     auto f = json::object();
        //     f["IsRTV"] = false;
        //     f["IsMega"] = false;
        //     f["LocalVarsBuffersDecl"] = {};
        //     f["Name"] = i->getNameAsString();
        //     f["ReturnType"] = render_type_name(i->getReturnType());

        //     std::string decl;
        //     auto in_outs = json::array();
        //     for (size_t j = 0; j < i->getNumParams(); j++) {
        //         auto v = i->getParamDecl(j);
        //         decl += render_type_name(v->getType());
        //         decl += " ";
        //         decl += v->getNameAsString();
        //         if (j < i->getNumParams() - 1)
        //             decl += ", ";
        //         if (v->getType()->isPointerType()) {
        //             auto var = json::object();
        //             var["IsTexture"] = false;
        //             var["Name"] = v->getNameAsString();
        //             in_outs.push_back(var);
        //         }
        //     }
        //     f["DeclOrig"] = i->getNameAsString() + "(" + decl + ")";
        //     f["Decl"] = i->getNameAsString() + "Cmd(VkCommandBuffer a_commandBuffer, " + decl + ")";
        //     f["MainFuncDeclCmd"] = f["Decl"];
        //     f["InOutVars"] = in_outs;
        //     f["OverrideMe"] = true;
        //     f["MainFuncTextCmd"] = "CODE_IS_HERE();";

        //     f["FullImpl"] = json::object();
        //     f["FullImpl"]["InputData"] = json::array();
        //     f["FullImpl"]["OutputData"] = json::array();
        //     std::string in_out = "";
        //     for (size_t j = 0; j < i->getNumParams(); j++) {
        //         auto p = i->getParamDecl(j);
        //         if (p->getType()->isPointerType()) {
        //             auto var = json::object();
        //             var["IsTexture"] = false;
        //             var["Name"] = p->getNameAsString();
        //             var["DataSize"] = "width * height";
        //             var["DataType"] = render_type_name(p->getType()->getPointeeType());
        //             f["FullImpl"]["OutputData"].push_back(var);
        //             in_out += p->getNameAsString() + "Buffer" + ", 0, ";
        //         }
        //     }
        //     f["FullImpl"]["ArgsOnSetInOut"] = in_out + "0";
        //     f["FullImpl"]["HasImages"] = false;
        //     {
        //         std::string args;
        //         for (size_t j = 0; j < i->getNumParams(); j++) {
        //             args += i->getParamDecl(j)->getNameAsString();
        //             if (j < i->getNumParams() - 1)
        //                 args += ", ";
        //         }
        //         f["FullImpl"]["ArgsOnCall"] = args;
        //     }

        //     f["DescriptorSets"] = json::array();

        //     for (auto kernel : controls_kernel_calls[i]) {
        //         auto ds = json::object();
        //         ds["Id"] = ds_id++;
        //         ds["KernelName"] = kernel->getNameAsString();
        //         ds["ArgNames"] = "ARG_NAMES()";
        //         ds["IsServiceCall"] = false;
        //         ds["IsVirtual"] = false;
        //         ds["ArgNumber"] = kernels_bindings[kernel].size();
        //         ds["Args"] = json::array();
        //         // outs() << "size here: " << kernels_bindings[kernel].size() << "\n";
        //         size_t index = 0;
        //         for (auto binding : kernels_bindings[kernel]) {
        //             auto arg = json::object();
        //             arg["IsTexture"] = false;
        //             arg["IsTextureArray"] = false;
        //             arg["IsAccelStruct"] = false;
        //             arg["Id"] = index++;
        //             arg["Name"] = binding->getNameAsString();
        //             desc_count++;
        //             ds["Args"].push_back(arg);
        //         }
        //         // outs() << ds["Args"].size() << "\n";
        //         f["DescriptorSets"].push_back(ds);
        //     }

        //     data["MainFunctions"].push_back(f);
        // }
        // data["TotalBuffersUsed"] = desc_count;

        data["UsePipelineCache"] = false;
        data["UseSpecConstWgSize"] = false;
        data["Hierarchies"] = json::array();
        data["UseServiceMemCopy"] = false;
        data["UseMatMult"] = false;
        data["IsMega"] = false;
        data["IsRTV"] = false;
        data["UniformUBO"] = false;
        data["ContantUBO"] = true;
        data["ISV2"] = json::array();
        data["TextureMembers"] = json::array();
        data["GlobalUseInt64"] = false;
        data["GlobalUseFloat64"] = false;
        data["GlobalUseInt16"] = false;
        data["HasRTXAccelStruct"] = false;
        data["ForceRayGen"] = false;
        data["UseCallable"] = false;
        data["HasIntersectionShaders"] = false;
        data["HasVarPointers"] = false;
        data["HasTextureArray"] = false;
        data["GlobalUseDoubleAtomics"] = false;
        data["GlobalUseFloatAtomics"] = false;
        data["GlobalUseHalf"] = false;
        data["GlobalUseInt8"] = false;
        data["GlobalUse8BitStorage"] = false;
        data["GlobalUse16BitStorage"] = false;

        // data["TotalBuffersUsed"] = 95; // 17 * 5 // WHY????
        data["TotalTexArrayUsed"] = 0;
        data["TotalTexCombinedUsed"] = 0;
        data["TotalTexStorageUsed"] = 0;
        data["TotalAccels"] = 0;
        // data["TotalDSNumber"] = 13; // WHY???

        data["MainInclude"] = "D:/gml_private/VolumeRenderer/VolumeRenderer.h";
        data["GenGpuApi"] = false;
        data["AdditionalIncludes"] = json::array();

        data["ClassDecls"] = json::array();

        // data["UBO"] = json::object();
        // data["UBO"]["UBOStructFields"] = json::array();
        // for (auto i : uniform_fields) {
        //     auto field = json::object();
        //     field["Name"] = i->getNameAsString();
        //     field["Type"] = render_type_name(i->getType());
        //     field["IsArray"] = get_type_array_size(i->getType()) > 0;
        //     field["ArraySize"] = get_type_array_size(i->getType());
        //     field["IsDummy"] = false;
        //     data["UBO"]["UBOStructFields"].push_back(field);
        // }

        // data["ClassVars"] = json::array();
        // for (auto i : uniform_fields) {
        //     auto var = json::object();
        //     var["IsArray"] = i->getType()->isArrayType();
        //     var["HasPrefix"] = false;
        //     var["IsConst"] = false;
        //     var["Name"] = i->getNameAsString();
        //     data["ClassVars"].push_back(var);
        // }

        data["SettersDecl"] = json::array();
        data["HasNameFunc"] = true;
        data["SetterFuncs"] = json::array();
        data["UpdateVectorFun"] = json::array();
        data["HasPrefixData"] = false;
        data["HasCommitDeviceFunc"] = true;
        data["HasGetTimeFunc"] = true;
        data["UpdateMembersPlainData"] = true;
        data["UpdateMembersVectorData"] = true;
        data["UpdateMembersTextureData"] = false;
        data["HasGetResDirFunc"] = false;
        data["ShaderFolderPrefix"] = "VolumeRenderer/";
        data["PlainMembersUpdateFunctions"] = json::array();
        data["VectorMembersUpdateFunctions"] = json::array();
        data["SetterVars"] = json::array();
        data["DefineGetTempBufferSize"] = false;
        data["GenerateSceneRestrictions"] = true;

        // kslicer::ApplyJsonToTemplate("templates/vk_class_init.cpp", fullSuffix + "_init.cpp", jsonHost);
        inja::Environment env;
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        {
            inja::Template t = env.parse_template("templates/vk_class.h");
            std::string result = env.render(t, data);

            std::ofstream out("class_slang.h");
            out << result << std::endl;
        }
        {
            inja::Template t = env.parse_template("templates/vk_class.cpp");
            std::string result = env.render(t, data);

            std::ofstream out("class_slang.cpp");
            out << result << std::endl;
        }
        {
            inja::Template t = env.parse_template("templates/vk_class_init.cpp");
            std::string result = env.render(t, data);

            std::ofstream out("class_slang_init.cpp");
            out << result << std::endl;
        }
        {
            inja::Template t = env.parse_template("templates/vk_class_ds.cpp");
            std::string result = env.render(t, data);

            std::ofstream out("class_slang_ds.cpp");
            out << result << std::endl;
        }
    }

} // namespace common_rewriter
