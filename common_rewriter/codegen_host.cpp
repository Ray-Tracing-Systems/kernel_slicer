#include "codegen_host.h"
#include "class.h"
#include "utils.h"
#include <clang/AST/AST.h>

namespace common_rewriter {
    using namespace nlohmann;

    void codegen() {

        json data = json::object();

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

        data["ClassVectorVars"] = json::array();
        for (auto i : buffer_fields) {
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
        for (auto i : buffer_fields) {
            auto buffer = json::object();
            buffer["Name"] = i->getNameAsString();
            buffer["IsVFHBuffer"] = false;
            buffer["VFHLevel"] = 0;
            buffer["HasPrefix"] = false;
            data["VectorMembers"].push_back(buffer);
        }

        data["TexArrayMembers"] = json::array();

        data["ClassTextureVars"] = json::array();
        data["ClassTexArrayVars"] = json::array();
        data["SamplerMembers"] = json::array();
        data["RedVectorVars"] = json::array(); // ???
        data["IndirectDispatches"] = json::array();

        data["MultipleSourceShaders"] = true;
        data["ShaderFolder"] = "./shaders_slang/";

        data["Kernels"] = json::array();
        for (auto i : kernels) {
            auto kernel = json::object();
            kernel["Name"] = i->getNameAsString().substr(9);
            kernel["OriginalName"] = i->getNameAsString();
            kernel["IsMega"] = false;
            kernel["UseRayGen"] = false;
            kernel["FinishRed"] = false;
            kernel["HasLoopInit"] = false;
            kernel["HasLoopFinish"] = false;
            kernel["IsIndirect"] = false;

            {
                std::string decl;
                // decl += "virtual ";
                // decl += render_type_name(i->getReturnType());
                // decl += " ";
                decl += i->getNameAsString().substr(9) + "Cmd";
                decl += "(";
                for (size_t j = 0; j < i->getNumParams(); j++) {
                    decl += render_type_name(i->getParamDecl(j)->getType());
                    decl += " ";
                    decl += i->getParamDecl(j)->getNameAsString();
                    if (j < i->getNumParams() - 1)
                        decl += ", ";
                }
                decl += ")";
                kernel["Decl"] = decl;
            }

            if (i->getNameAsString().substr(6, 1) == "1") {
                kernel["WGSizeX"] = 256;
                kernel["WGSizeY"] = 1;
                kernel["WGSizeZ"] = 1;
            } else if (i->getNameAsString().substr(6, 1) == "2") {
                kernel["WGSizeX"] = 32;
                kernel["WGSizeY"] = 8;
                kernel["WGSizeZ"] = 1;
            } else if (i->getNameAsString().substr(6, 1) == "3") {
                kernel["WGSizeX"] = 8;
                kernel["WGSizeY"] = 8;
                kernel["WGSizeZ"] = 8;
            } else {
                REWRITE_ASSERT(false);
            }

            data["Kernels"].push_back(kernel);
        }

        data["kernelsDecls"] = json::array();
        for (auto i : kernels) {
            std::string decl;
            decl += "virtual ";
            decl += render_type_name(i->getReturnType());
            decl += " ";
            decl += i->getNameAsString().substr(9) + "Cmd";
            decl += "(";
            for (size_t j = 0; j < i->getNumParams(); j++) {
                decl += render_type_name(i->getParamDecl(j)->getType());
                decl += " ";
                decl += i->getParamDecl(j)->getNameAsString();
                if (j < i->getNumParams() - 1)
                    decl += ", ";
            }
            decl += ");";
            data["KernelsDecls"].push_back(decl);
        }

        data["MainFunctions"] = json::array();

        for (auto i : controls) {
            auto f = json::object();
            f["IsRTV"] = false;
            f["IsMega"] = false;
            f["LocalVarsBuffersDecl"] = {};
            f["Name"] = i->getNameAsString();
            f["ReturnType"] = render_type_name(i->getReturnType());

            std::string decl;
            auto in_outs = json::array();
            for (size_t j = 0; j < i->getNumParams(); j++) {
                auto v = i->getParamDecl(j);
                decl += render_type_name(v->getType());
                decl += " ";
                decl += v->getNameAsString();
                if (j < i->getNumParams() - 1)
                    decl += ", ";
                if (v->getType()->isPointerType()) {
                    auto var = json::object();
                    var["IsTexture"] = false;
                    var["Name"] = v->getNameAsString();
                    in_outs.push_back(var);
                }
            }
            f["DeclOrig"] = i->getNameAsString() + "(" + decl + ")";
            f["Decl"] = i->getNameAsString() + "Cmd(VkCommandBuffer a_commandBuffer, " + decl + ")";
            f["InOutVars"] = in_outs;
            f["OverrideMe"] = true;
            data["MainFunctions"].push_back(f);
        }

        data["UsePipelineCache"] = false;
        data["UseSpecConstWgSize"] = false;
        data["Hierarchies"] = json::array();
        data["UseServiceMemCopy"] = false;
        data["UseMatMult"] = false;
        data["IsMega"] = false;
        data["IsRTV"] = false;
        data["UniformUBO"] = false;
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

        data["TotalBuffersUsed"] = 95; // 17 * 5 // WHY????
        data["TotalTexArrayUsed"] = 0;
        data["TotalTexCombinedUsed"] = 0;
        data["TotalTexStorageUsed"] = 0;
        data["TotalAccels"] = 0;
        data["TotalDSNumber"] = 13; // WHY???

        data["MainInclude"] = "D:/gml_private/VolumeRenderer/VolumeRenderer.h";
        data["GenGpuApi"] = false;
        data["AdditionalIncludes"] = json::array();

        data["ClassDecls"] = json::array();

        data["UBO"] = json::object();
        data["UBO"]["UBOStructFields"] = json::array();
        for (auto i : uniform_fields) {
            auto field = json::object();
            field["Name"] = i->getNameAsString();
            field["Type"] = render_type_name(i->getType());
            field["IsArray"] = get_type_array_size(i->getType()) > 0;
            field["ArraySize"] = get_type_array_size(i->getType());
            field["IsDummy"] = false;
            data["UBO"]["UBOStructFields"].push_back(field);
        }

        data["ClassVars"] = json::array();
        for (auto i : uniform_fields) {
            auto var = json::object();
            var["IsArray"] = i->getType()->isArrayType();
            var["HasPrefix"] = false;
            var["IsConst"] = false;
            var["Name"] = i->getNameAsString();
            data["ClassVars"].push_back(var);
        }

        data["ClassVectorVars"] = json::array();
        for (auto i : buffer_fields) {
            auto var = json::object();
            var["Name"] = i->getNameAsString();
            var["AccessSymb"] = ".";
            var["IsVFHBuffer"] = false;
            var["VFHLevel"] = 0;
            var["TypeOfData"] = render_type_name(get_vector_specialization_type(i->getType()));
            data["ClassVectorVars"].push_back(var);
        }

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
        // {
        //     inja::Template t = env.parse_template("templates/vk_class_ds.cpp");
        //     std::string result = env.render(t, data);

        //     std::ofstream out("class_slang_ds.cpp");
        //     out << result << std::endl;
        // }
    }

} // namespace common_rewriter
