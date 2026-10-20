#include "codegen_host.h"
// #include "class.h"
#include "class_info.h"
#include "codegen.h"
#include "errors.h"
#include "utils.h"
#include "clang/AST/Attr.h"
#include <clang/AST/AST.h>
#include <inja.hpp>
namespace kslicer {
    using namespace nlohmann;
    using llvm::outs;

    void codegen(const Config& config, const ClassInfo& info) {

        auto base = (config.config_path.parent_path() / config.base_directory).lexically_normal();
        auto class_directory = (base / config.sources[0]).parent_path().lexically_normal();

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
                KSLICER_ACCERT(false);
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
            kernel["EnableBlockExpansion"] = false;

            auto args = json::array();
            for (auto binding : i.bindings) {
                auto arg = json::object();
                arg["Name"] = "m_vdata" + binding->getNameAsString();
                arg["Id"] = args.size();
                arg["Type"] = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER";
                arg["IsTextureArray"] = false;
                arg["Count"] = 1;
                args.push_back(arg);
            }
            kernel["Args"] = args;
            kernel["ArgCount"] = args.size();
            kernel["IsRTV"] = false;

            {
                KernelCodegen codegen(info, i.decl);

                for (auto i : info.used_structs) {
                    codegen.buffer += "struct " + i->getNameAsString() + "\n{\n";
                    for (auto field : i->fields()) {
                        codegen.buffer += "    ";
                        codegen.render_type(field->getType());
                        codegen.buffer += " ";
                        codegen.buffer += field->getNameAsString();
                        if (field->getType()->isArrayType()) {
                            codegen.buffer += "[" + std::to_string(get_type_array_size(field->getType())) + "]";
                        }
                        codegen.buffer += ";\n";
                    }
                    codegen.buffer += "};\n";
                }

                codegen.buffer += "struct UBO\n{\n";

                for (auto i : info.used_uniform_fields) {
                    codegen.buffer += "    ";
                    codegen.render_type(i->getType());
                    codegen.buffer += " ";
                    codegen.buffer += i->getNameAsString();
                    codegen.buffer += ";\n";
                }

                for (auto i : info.used_buffer_fields) {
                    codegen.buffer += "    uint " + i->getNameAsString() + "_size;\n";
                    codegen.buffer += "    uint " + i->getNameAsString() + "_capacity;\n";
                }

                codegen.buffer += "    uint dummy_last;\n";
                codegen.buffer += "};\n";

                for (auto binding : i.bindings) {
                    QualType t = binding->getType();
                    QualType u = t->isPointerType() ? t->getPointeeType() : get_vector_specialization_type(t);
                    codegen.buffer +=
                        "RWStructuredBuffer<" + render_type_name(u) + "> " + binding->getNameAsString() + ";\n";
                }
                codegen.buffer += "StructuredBuffer<UBO> ubo;\n";

                codegen.buffer += R"(                
float4   operator*(float4x4 m, float4 v) { return mul(m,v); }
float3   operator*(float4x4 m, float3 v) { return mul(m, float4(v,1.0f)).xyz; }
float3   operator*(float3x3 m, float3 v) { return mul(m,v); }
)";

                for (auto i : info.used_global_constants) {
                    codegen.buffer += "static const " + render_type_name(i->getType()) + " " + i->getNameAsString();
                    if (i->getType()->isArrayType()) {
                        codegen.buffer += "[" + std::to_string(get_type_array_size(i->getType())) + "]";
                    }
                    codegen.buffer += " = ";
                    codegen.render_expression(i->getInit());
                    codegen.buffer += ";\n";
                }

                for (auto i : info.used_functions) {
                    codegen.render_function(i);
                }

                codegen.buffer += "struct KernelArgs\n{\n";

                for (auto j : kernel["AuxArgs"]) {
                    codegen.buffer +=
                        "    " + j["Type"].get<std::string>() + " " + j["Name"].get<std::string>() + ";\n";
                }

                for (auto j : i.bounds_args) {
                    codegen.buffer += "    uint " + j->getNameAsString() + ";\n";
                }

                if (i.bounds_args.size() < 2) {
                    codegen.buffer += "    uint iNumElementsY;\n";
                }
                if (i.bounds_args.size() < 3) {
                    codegen.buffer += "    uint iNumElementsZ;\n";
                }
                codegen.buffer += "    uint tFlagsMask;\n";

                codegen.buffer += "};\n";

                codegen.buffer += "[shader(\"compute\")]\n";
                codegen.buffer += "[numthreads(" + std::to_string(kernel["WGSizeX"].get<size_t>()) + ", " +
                                  std::to_string(kernel["WGSizeY"].get<size_t>()) + ", " +
                                  std::to_string(kernel["WGSizeZ"].get<size_t>()) + ")]\n";
                codegen.buffer += "void main(uint3 a_globalTID: SV_DispatchThreadID, uint3 a_localTID : "
                                  "SV_GroupThreadID, uniform KernelArgs kgenArgs)\n";
                // for (auto index : i.index_vars) {
                //     codegen.buffer += index->getNameAsString() + "\n";
                // }

                codegen.buffer += "{\n";

                for (size_t j = 0; j < i.index_vars.size(); j++) {
                    codegen.buffer += "    uint " + i.index_vars[j]->getNameAsString() + " = uint(a_globalTID[" +
                                      std::to_string(j) + "]);\n";
                }

                for (size_t j = 0; j < i.decl->getNumParams(); j++) {
                    auto p = i.decl->getParamDecl(j);
                    if (!p->getType()->isPointerType()) {
                        codegen.buffer += "    ";
                        codegen.buffer += render_type_name(p->getType());
                        codegen.buffer += " ";
                        codegen.buffer += p->getNameAsString();
                        codegen.buffer += " = ";
                        codegen.buffer += "kgenArgs.";
                        codegen.buffer += p->getNameAsString();
                        codegen.buffer += ";\n";
                    }
                }

                codegen.buffer += "    if (";
                for (size_t j = 0; j < i.index_vars.size(); j++) {
                    codegen.buffer += i.index_vars[j]->getNameAsString() + " < " + i.bounds_args[j]->getNameAsString();
                    if (j < i.index_vars.size() - 1) {
                        codegen.buffer += " && ";
                    }
                }
                codegen.buffer += ")\n";
                codegen.render_statement(i.body, 1);
                codegen.buffer += "}\n";
                std::ofstream file(base / config.shaders_directory / ("shaders" + config.suffix) /
                                   (i.decl->getNameAsString() + ".slang"));
                file << codegen.buffer;
            }

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
                    bool found_size_annotation = false;
                    std::string size_value;
                    for (const AnnotateAttr* attr : p->specific_attrs<AnnotateAttr>()) {
                        if (attr->getAnnotation() == StringRef{"kslicer::size"}) {
                            KSLICER_ACCERT(attr->args_size() == 1);
                            const StringLiteral* value = dyn_cast<StringLiteral>(*attr->args_begin());
                            KSLICER_ACCERT(value);
                            std::string text = string_from_source_range(value->getSourceRange());
                            text = text.substr(1, text.length() - 2);
                            outs() << i.decl->getNameAsString() << ": " << p->getNameAsString() << ".size = " << text
                                   << "\n";
                            size_value = text;
                            found_size_annotation = true;
                        }
                    }

                    if (!found_size_annotation) {
                        error_missing_size_annotation(p);
                    }

                    auto var = json::object();
                    var["IsTexture"] = false;
                    var["Name"] = p->getNameAsString();
                    var["DataSize"] = size_value;
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
                // ds["ArgNumber"] = kernel->bindings.size();
                ds["Layout"] = kernel->decl->getNameAsString().substr(9) + "DSLayout";
                ds["Args"] = json::array();

                size_t index = 0;

                for (size_t j = 0; j < call_expr->getNumArgs(); j++) {
                    auto arg = call_expr->getArg(j)->IgnoreParenImpCasts();
                    if (auto mcall = dyn_cast<CXXMemberCallExpr>(arg)) {
                        auto obj = mcall->getImplicitObjectArgument()->IgnoreParenImpCasts();
                        if (auto member = dyn_cast<MemberExpr>(obj)) {
                            if (dyn_cast<CXXThisExpr>(member->getBase())) {
                                if (mcall->getMethodDecl()->getNameAsString() == "data") {
                                    // note_found(mcall->getExprLoc(), "method call");
                                    // member->getMemberDecl()
                                    auto arg = json::object();
                                    arg["IsTexture"] = false;
                                    arg["IsTextureArray"] = false;
                                    arg["IsAccelStruct"] = false;
                                    arg["Id"] = index++;
                                    arg["Name"] = "m_vdata." + member->getMemberDecl()->getNameAsString();
                                    descriptors_count++;
                                    ds["Args"].push_back(arg);
                                }
                            }
                        }
                    } else if (auto ref = dyn_cast<DeclRefExpr>(arg)) {
                        if (auto p = dyn_cast<ParmVarDecl>(ref->getDecl())) {
                            if (p->getType()->isPointerType()) {
                                // note_found(p->getLocation(), "uses arg");
                                auto arg = json::object();
                                arg["IsTexture"] = false;
                                arg["IsTextureArray"] = false;
                                arg["IsAccelStruct"] = false;
                                arg["Id"] = index++;
                                arg["Name"] = i.decl->getNameAsString() + "_local." + p->getNameAsString();
                                descriptors_count++;
                                ds["Args"].push_back(arg);
                            }
                        }
                    }
                }

                for (auto binding : kernel->bindings) {
                    if (!dyn_cast<FieldDecl>(binding)) {
                        continue;
                    }
                    auto arg = json::object();
                    arg["IsTexture"] = false;
                    arg["IsTextureArray"] = false;
                    arg["IsAccelStruct"] = false;
                    arg["Id"] = index++;
                    arg["Name"] = "m_vdata." + binding->getNameAsString();
                    descriptors_count++;
                    ds["Args"].push_back(arg);
                }
                // outs() << ds["Args"].size() << "\n";
                ds["ArgNumber"] = index;
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
        for (auto i : info.used_buffer_fields) {
            auto field = json::object();
            field["Name"] = i->getNameAsString() + "_size";
            field["Type"] = "uint";
            field["IsArray"] = false;
            field["ArraySize"] = 0;
            field["IsDummy"] = false;
            data["UBO"]["UBOStructFields"].push_back(field);
            field["Name"] = i->getNameAsString() + "_capacity";
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
            buffer["IsHashMap"] = false;
            data["ClassVectorVars"].push_back(buffer);
        }

        data["VectorMembers"] = json::array();
        for (auto i : info.used_buffer_fields) {
            auto buffer = json::object();
            buffer["Name"] = i->getNameAsString();
            buffer["IsVFHBuffer"] = false;
            buffer["VFHLevel"] = 0;
            buffer["HasPrefix"] = false;
            buffer["IsHashMap"] = false;
            data["VectorMembers"].push_back(buffer);
        }

        data["UseServiceScan"] = false;
        data["UseServiceSort"] = false;
        data["HasAllRefs"] = false;
        data["UseRayGen"] = false;
        data["IncludeClassDecl"] = config.main_class + config.suffix + ".h";
        data["SceneMembers"] = json::array();
        data["ForceProxy"] = false;
        data["IntersectionHierarhcy"] = json::object();
        data["IntersectionHierarhcy"]["Implementations"] = json::array();
        data["HasIntersectionShaders"] = false;
        data["Constructors"] = json::array();
        {
            auto ctor = json::object();
            ctor["NumParams"] = 0;
            ctor["ClassName"] = config.main_class;
            data["Constructors"].push_back(ctor);
        }

        data["MainClassName"] = config.main_class;
        data["MainClassSuffix"] = config.suffix;

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
        data["ShaderFolder"] = "./shaders" + config.suffix + "/";

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
        //         KSLICER_ACCERT(false);
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

        data["MainInclude"] = config.main_class + ".h";
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
        data["ShaderFolderPrefix"] = config.shaders_directory.string();
        data["PlainMembersUpdateFunctions"] = json::array();
        data["VectorMembersUpdateFunctions"] = json::array();
        data["SetterVars"] = json::array();
        data["DefineGetTempBufferSize"] = false;
        data["GenerateSceneRestrictions"] = true;

        data["HasHashMaps"] = false;
        data["HashMapSubgroups"] = false;

        inja::Environment env;
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        {
            inja::Template t = env.parse_template((config.kslicer_directory / "templates/vk_class.h").string());
            std::string result = env.render(t, data);

            std::ofstream out(class_directory / (config.main_class + config.suffix + ".h"));
            out << result << std::endl;
        }
        {
            inja::Template t = env.parse_template((config.kslicer_directory / "templates/vk_class.cpp").string());
            std::string result = env.render(t, data);

            std::ofstream out(class_directory / (config.main_class + config.suffix + ".cpp"));
            out << result << std::endl;
        }
        {
            inja::Template t = env.parse_template((config.kslicer_directory / "templates/vk_class_init.cpp").string());
            std::string result = env.render(t, data);

            std::ofstream out(class_directory / (config.main_class + config.suffix + "_init.cpp"));
            out << result << std::endl;
        }
        {
            inja::Template t = env.parse_template((config.kslicer_directory / "templates/vk_class_ds.cpp").string());
            std::string result = env.render(t, data);

            std::ofstream out(class_directory / (config.main_class + config.suffix + "_ds.cpp"));
            out << result << std::endl;
        }
    }

} // namespace kslicer
