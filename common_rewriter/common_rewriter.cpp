#include "common_rewriter.h"
#include <clang/Frontend/CompilerInstance.h>
#include <clang/AST/ASTContext.h>
#include <clang/Lex/Lexer.h>
#include <fstream>

#define REWRITE_ASSERT(cond) \
    (!!(cond) ? (void)0 : (fprintf(stderr, "Assertion failed: %s\n", #cond), abort()));

namespace common_rewriter
{
    // https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
    static bool ends_with(std::string_view str, std::string_view suffix)
    {
        return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static bool starts_with(std::string_view str, std::string_view prefix)
    {
        return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
    }

    using namespace clang;
    using namespace llvm;

    bool inside_kernel_ = false;

    const CompilerInstance *compiler_instance;

    std::vector<const CXXMethodDecl *> called_methods;

    std::map<const FunctionDecl *, std::string> functions;

    std::string builtin_functions[] = {
        "min", "max", "clamp", "floor", "abs", "dot", "length", "normalize", "exp", "sqrt", "copysign"};

    void add_method(const CXXMethodDecl *method)
    {
        method = dyn_cast<CXXMethodDecl>(method->getDefinition());
        if (std::find(called_methods.begin(), called_methods.end(), method) == called_methods.end())
            called_methods.push_back(method);
    }

    std::vector<const FunctionDecl *> called_functions;

    void add_function(const FunctionDecl *function)
    {
        if (std::find(std::begin(builtin_functions), std::end(builtin_functions), function->getNameAsString()) == std::end(builtin_functions))
        {
            REWRITE_ASSERT(function->getDefinition());
            function = function->getDefinition();
            if (std::find(called_functions.begin(), called_functions.end(), function) == called_functions.end())
                called_functions.push_back(function);
        }
    }

    std::string buffer_;
    bool do_emit_ = false;

    void emit(std::string s)
    {
        if (do_emit_)
            buffer_ += s;
    }

    void emit_indent(size_t indent)
    {
        for (size_t i = 0; i < indent; i++)
            emit("    ");
    }

    void emit(const char *s)
    {
        emit(std::string(s));
    }

    void emit(const llvm::StringRef &str)
    {
        emit(std::string(str));
    }

    const ASTContext &context()
    {
        return compiler_instance->getASTContext();
    }

    std::string string_from_source_range(SourceRange range)
    {
        // https://stackoverflow.com/questions/11083066/getting-the-source-behind-clangs-ast
        const SourceManager &source_manager = compiler_instance->getSourceManager();
        clang::SourceLocation true_end(clang::Lexer::getLocForEndOfToken(range.getEnd(), 0, source_manager, {}));
        return std::string(source_manager.getCharacterData(range.getBegin()), source_manager.getCharacterData(true_end) - source_manager.getCharacterData(range.getBegin()));
    }

    void note_found(SourceLocation loc, std::string what)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Remark, "Found %0");
        diagnostic_engine.Report(loc, id) << what;
    }

    void error_unknown_stmt_class(const Stmt *stmt)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Unknown statement kind: %0");
        diagnostic_engine.Report(stmt->getBeginLoc(), id) << stmt->getStmtClassName();
    }

    void error_expr_needs_cleanups(const ExprWithCleanups *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Expression needs cleanups");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_unknown_overloaded_operator(const CXXOperatorCallExpr *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Unknown operator overload '%0'");
        diagnostic_engine.Report(expr->getExprLoc(), id) << getOperatorSpelling(expr->getOperator());
    }

    void error_can_only_access_locals(const Expr *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Trying to access non local variable");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_can_only_call_methods_of_this(const Expr *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Can only call methods of this");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_multiple_declarations(const DeclStmt *stmt)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Multiple declarations are not supported for now");
        diagnostic_engine.Report(stmt->getBeginLoc(), id);
    }

    NamedDecl *find_decl_by_name(std::string name)
    {
        TranslationUnitDecl *translation_unit_declaration_context = context().getTranslationUnitDecl();
        IdentifierInfo &identifier_info = context().Idents.get(name);
        DeclContext::lookup_result res = translation_unit_declaration_context->lookup(&identifier_info);

        if (res.empty())
        {
            outs() << name << " is not found\n";
            exit(1);
        }
        if (!res.isSingleResult())
        {
            outs() << name << " is not unique\n";
            exit(1);
        }
        return res.front();
    }

    void rewrite_type(QualType type)
    {
        // dyn_cast<ReferenceType>(type.getTypePtr())
        std::string s;
        llvm::raw_string_ostream ss(s);
        type.print(ss, {{}});

        // if (ends_with(s, "float2"))
        //     emit("float2");
        // else if (ends_with(s, "float3"))
        //     emit("float3");
        // else if (ends_with(s, "float4"))
        //     emit("float4");
        // else
        emit(s);
    }

    void rewrite_expression(const Expr *expr_)
    {
        if (auto expr = dyn_cast<ExprWithCleanups>(expr_))
        {
            // Lets not do destroying of temporary variables
            if (expr->getNumObjects() > 0)
                error_expr_needs_cleanups(expr);
            rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<FullExpr>(expr_))
        {
            rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<ImplicitCastExpr>(expr_))
        {
            // Ignoring implicit casts
            return rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<MaterializeTemporaryExpr>(expr_))
        {
            // I don't know what it is...
            return rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<ParenExpr>(expr_))
        {
            emit("(");
            rewrite_expression(expr->getSubExpr());
            emit(")");
        }
        else if (auto expr = dyn_cast<CStyleCastExpr>(expr_))
        {
            emit("(");
            rewrite_type(expr->getType());
            emit(")");
            rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<CXXFunctionalCastExpr>(expr_))
        {
            rewrite_type(expr->getType());
            emit("(");
            rewrite_expression(expr->getSubExpr());
            emit(")");
        }
        else if (auto expr = dyn_cast<CXXConstructExpr>(expr_))
        {
            rewrite_type(expr->getType());
            emit("(");
            for (size_t i = 0; i < expr->getNumArgs(); i++)
            {
                rewrite_expression(expr->getArg(i));
                if (i < expr->getNumArgs() - 1)
                    emit(", ");
            }
            emit(")");
        }
        else if (auto expr = dyn_cast<BinaryOperator>(expr_))
        {
            rewrite_expression(expr->getLHS());
            emit(" ");
            emit(BinaryOperator::getOpcodeStr(expr->getOpcode()));
            emit(" ");
            rewrite_expression(expr->getRHS());
        }
        else if (auto expr = dyn_cast<UnaryOperator>(expr_))
        {
            emit(UnaryOperator::getOpcodeStr(expr->getOpcode()));
            rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<CXXOperatorCallExpr>(expr_))
        {
            // note_found(expr->getExprLoc(), "here");
            if (expr->getNumArgs() == 1)
            {
                std::string s = getOperatorSpelling(expr->getOperator());
                if (s == "+" || s == "-")
                {
                    emit(s);
                    rewrite_expression(expr->getArg(0));
                }
                else
                {
                    error_unknown_overloaded_operator(expr);
                }
            }
            else
            {
                REWRITE_ASSERT(expr->getNumArgs() == 2);
                std::string s = getOperatorSpelling(expr->getOperator());

                std::string known_binary_operators[] = {
                    "=",
                    "+",
                    "-",
                    "*",
                    "/",
                    "+=",
                    "-=",
                    "*=",
                    "/=",
                    "&",
                    "|",
                    "<<",
                    ">>"};

                if (std::find(std::begin(known_binary_operators), std::end(known_binary_operators), s) != std::end(known_binary_operators))
                {
                    rewrite_expression(expr->getArg(0));
                    emit(" ");
                    emit(s);
                    emit(" ");
                    rewrite_expression(expr->getArg(1));
                }
                else if (s == "[]")
                {
                    rewrite_expression(expr->getArg(0));
                    emit("[");
                    rewrite_expression(expr->getArg(1));
                    emit("]");
                }
                else
                {
                    error_unknown_overloaded_operator(expr);
                }
            }
        }
        else if (auto expr = dyn_cast<ArraySubscriptExpr>(expr_))
        {
            rewrite_expression(expr->getBase());
            emit("[");
            rewrite_expression(expr->getIdx());
            emit("]");
        }
        else if (auto expr = dyn_cast<ConditionalOperator>(expr_))
        {
            rewrite_expression(expr->getCond());
            emit(" ? ");
            rewrite_expression(expr->getTrueExpr());
            emit(" : ");
            rewrite_expression(expr->getFalseExpr());
        }
        else if (auto expr = dyn_cast<IntegerLiteral>(expr_))
        {
            emit(string_from_source_range(expr->getSourceRange()));
        }
        else if (auto expr = dyn_cast<FloatingLiteral>(expr_))
        {
            emit(string_from_source_range(expr->getSourceRange()));
        }
        else if (auto expr = dyn_cast<CXXBoolLiteralExpr>(expr_))
        {
            emit(string_from_source_range(expr->getSourceRange()));
        }
        else if (auto expr = dyn_cast<DeclRefExpr>(expr_))
        {
            auto decl = expr->getDecl();
            if (inside_kernel_)
            {
                auto param = dyn_cast<ParmVarDecl>(decl);
                if (param)
                {
                    emit("kgenArgs.");
                }
                // outs() << decl->getNameAsString() << ": " << param << "\n";
                emit(decl->getNameAsString());
            }
            else
            {
                emit(decl->getNameAsString());
            }
        }
        else if (auto expr = dyn_cast<MemberExpr>(expr_))
        {
            if (dyn_cast<CXXThisExpr>(expr->getBase()))
            {
                if (expr->getMemberDecl()->getType().getAsString({{}}).find("vector") != std::string::npos)
                {
                    emit(expr->getMemberDecl()->getNameAsString());
                }
                else
                {
                    emit("ubo[0].");
                    emit(expr->getMemberDecl()->getNameAsString());
                }
            }
            else
            {
                rewrite_expression(expr->getBase());
                // Accessing anonimous unions inside structs
                if (expr->getMemberDecl()->getNameAsString().length() > 0)
                {
                    emit(".");
                    emit(expr->getMemberDecl()->getNameAsString());
                }
            }
        }
        else if (auto expr = dyn_cast<CXXMemberCallExpr>(expr_))
        {

            if (expr->getMethodDecl()->getNameAsString() == "size")
            {
                rewrite_expression(expr->getImplicitObjectArgument());
                emit("_size");
            }
            else
            {

                if (!dyn_cast<CXXThisExpr>(expr->getImplicitObjectArgument()))
                    error_can_only_call_methods_of_this(expr);

                add_method(expr->getMethodDecl());
                emit(expr->getMethodDecl()->getNameAsString());
                emit("(");
                for (int i = 0; i < expr->getNumArgs(); i++)
                {
                    rewrite_expression(expr->getArg(i));
                    if (i < expr->getNumArgs() - 1)
                        emit(", ");
                }
                emit(")");
            }
        }
        else if (auto expr = dyn_cast<CallExpr>(expr_))
        {
            auto func = dyn_cast<FunctionDecl>(expr->getCalleeDecl());
            REWRITE_ASSERT(func);
            add_function(func);
            emit(func->getNameAsString());
            emit("(");
            for (int i = 0; i < expr->getNumArgs(); i++)
            {
                rewrite_expression(expr->getArg(i));
                if (i < expr->getNumArgs() - 1)
                    emit(", ");
            }
            emit(")");
        }
        else
        {
            error_unknown_stmt_class(expr_);
        }
    }

    void rewrite_statement(const Stmt *stmt_, size_t indent)
    {
        if (auto stmt = dyn_cast<CompoundStmt>(stmt_))
        {
            emit_indent(indent);
            emit("{\n");
            for (auto i : stmt->children())
            {
                rewrite_statement(i, indent + 1);
            }
            emit_indent(indent);
            emit("}\n");
        }
        else if (auto stmt = dyn_cast<Expr>(stmt_))
        {
            emit_indent(indent);
            rewrite_expression(stmt);
            emit(";\n");
        }
        else if (auto stmt = dyn_cast<DeclStmt>(stmt_))
        {

            for (auto i : stmt->getDeclGroup())
            {
                auto var = dyn_cast<VarDecl>(i);
                REWRITE_ASSERT(var);
                emit_indent(indent);
                rewrite_type(var->getType());
                emit(" ");
                emit(var->getNameAsString());
                if (var->getInit())
                {
                    emit(" = ");
                    rewrite_expression(var->getInit());
                }
                emit(";\n");
            }
        }
        else if (auto stmt = dyn_cast<IfStmt>(stmt_))
        {
            emit_indent(indent);
            emit("if (");
            rewrite_expression(stmt->getCond());
            emit(")\n");
            rewrite_statement(stmt->getThen(), indent);
            if (stmt->getElse())
            {
                emit_indent(indent);
                emit("else\n");
                rewrite_statement(stmt->getElse(), indent);
            }
        }
        else if (auto stmt = dyn_cast<WhileStmt>(stmt_))
        {
            emit_indent(indent);
            emit("while (");
            rewrite_expression(stmt->getCond());
            emit(")\n");
            rewrite_statement(stmt->getBody(), indent);
        }
        else if (auto stmt = dyn_cast<ForStmt>(stmt_))
        {
            emit_indent(indent);
            emit("for (");
            rewrite_statement(stmt->getInit(), 0);
            rewrite_expression(stmt->getCond());
            emit("; ");
            rewrite_expression(stmt->getInc());
            emit(")\n");
            rewrite_statement(stmt->getBody(), indent + 1);
        }
        else if (auto stmt = dyn_cast<BreakStmt>(stmt_))
        {
            emit_indent(indent);
            emit("break;\n");
        }
        else if (auto stmt = dyn_cast<ContinueStmt>(stmt_))
        {
            emit_indent(indent);
            emit("continue;\n");
        }
        else if (auto stmt = dyn_cast<ReturnStmt>(stmt_))
        {
            emit_indent(indent);
            emit("return ");
            if (stmt->getRetValue())
                rewrite_expression(stmt->getRetValue());
            emit(";\n");
        }
        else
        {
            error_unknown_stmt_class(stmt_);
        }
    }

    // void rewrite_method(const CXXMethodDecl *method)
    // {
    //     rewrite_type(method->getReturnType());
    //     emit(" ");
    //     emit(method->getNameAsString());
    //     emit("(");
    //     for (int i = 0; i < method->getNumParams(); i++)
    //     {
    //         auto decl = method->getParamDecl(i);
    //         rewrite_type(decl->getType());
    //         emit(" ");
    //         emit(decl->getNameAsString());
    //         if (i < method->getNumParams() - 1)
    //             emit(", ");
    //     }
    //     emit(")");
    //     rewrite_statement(method->getBody(), 0);
    // }

    void rewrite_function(const FunctionDecl *function)
    {
        buffer_ = "";
        rewrite_type(function->getReturnType());
        emit(" ");
        emit(function->getNameAsString());
        emit("(");
        for (int i = 0; i < function->getNumParams(); i++)
        {
            auto decl = function->getParamDecl(i);
            rewrite_type(decl->getType());
            emit(" ");
            emit(decl->getNameAsString());
            if (i < function->getNumParams() - 1)
                emit(", ");
        }
        emit(")");
        if (function->getBody())
        {
            rewrite_statement(function->getBody(), 0);
            functions[function] = buffer_;
        }
        else
        {
            emit(";\n");
        }
    }

    std::string slang_operators = R"(
        
float4 mul4x4x4(float4x4 m, float4 v) { return mul(m,v); }
float3 mul3x3  (float4x4 m, float3 v) { return to_float3(mul(m, to_float4(v, 0.0f))); }
float3 mul4x3  (float4x4 m, float3 v) { return to_float3(mul(m, to_float4(v, 1.0f))); }

float4   operator*(float4x4 m, float4 v) { return mul(m,v); }
float3   operator*(float4x4 m, float3 v) { return mul(m, float4(v,1.0f)).xyz; }
float3   operator*(float3x3 m, float3 v) { return mul(m,v); }
    )";

    std::string uniforms = R"(
        
RWStructuredBuffer<float4> out_color; // 
RWStructuredBuffer<float4> m_colorBuffer; // 
RWStructuredBuffer<float4> m_colored_grid; // 
RWStructuredBuffer<float> m_grid; // 
RWStructuredBuffer<uint> m_packedXY; // 
StructuredBuffer<VolumeRenderer_slang_UBO_Data> ubo;

    )";
    std::string ubo = R"(
        
struct VolumeRenderer_slang_UBO_Data
{
  float4x4 m_projInv;
  float4x4 m_worldViewInv;
  Plane m_cuttingPlane;
  float4 m_backgroundColor;
  float4 m_baseColor;
  SCom2Header m_header;
  VolumeRenderPreset m_preset;
  uint frame0;
  uint frame1;
  uint m_AAFrameNum;
  uint m_SdfDAGBrickTranspositionOffset;
  uint m_SdfDAGTransformCount;
  float m_dframe;
  uint m_grid_type;
  uint m_height;
  uint m_packedXY_height;
  uint m_packedXY_width;
  uint m_seed;
  uint m_sz;
  uint m_vox_cnt;
  uint m_width;
  uint m_DAGInverseTransforms_capacity;
  uint m_DAGInverseTransforms_size;
  uint m_RotAddTable_capacity;
  uint m_RotAddTable_size;
  uint m_SComNodes_capacity;
  uint m_SComNodes_size;
  uint m_SComValues_capacity;
  uint m_SComValues_size;
  uint m_SdfCompactOctreeRotModifiers_capacity;
  uint m_SdfCompactOctreeRotModifiers_size;
  uint m_SdfDAGChildEdges_capacity;
  uint m_SdfDAGChildEdges_size;
  uint m_SdfDAGDataEdges_capacity;
  uint m_SdfDAGDataEdges_size;
  uint m_SdfDAGDistances_capacity;
  uint m_SdfDAGDistances_size;
  uint m_SdfDAGHeaders_capacity;
  uint m_SdfDAGHeaders_size;
  uint m_SdfDAGNodes_capacity;
  uint m_SdfDAGNodes_size;
  uint m_SdfDAGTranspositions_capacity;
  uint m_SdfDAGTranspositions_size;
  uint m_SphericalHarmonics_capacity;
  uint m_SphericalHarmonics_size;
  uint m_colorBuffer_capacity;
  uint m_colorBuffer_size;
  uint m_colored_grid_capacity;
  uint m_colored_grid_size;
  uint m_grid_capacity;
  uint m_grid_size;
  uint m_packedXY_capacity;
  uint m_packedXY_size;
  uint m_timestamps_capacity;
  uint m_timestamps_size;
  uint dummy_last;
};
    )";

    std::string kernel_args = R"(
    struct KernelArgs
{
  uint sample_id;
  uint count; 
  uint iNumElementsY; 
  uint iNumElementsZ; 
  uint tFlagsMask;    
};
    )";

    std::string kernel_boiler = R"(
    
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 a_globalTID : SV_DispatchThreadID, uint3 a_localTID : SV_GroupThreadID, uniform KernelArgs kgenArgs)
{
    bool runThisThread = true;
    const uint tidX = uint(a_globalTID[0]);
    if (tidX >= kgenArgs.count + 0)
        runThisThread = false;
    // KERNEL BODY:
    if(runThisThread)
)";

    void rewrite_kernel(std::string class_name, std::string kernel_name)
    {

        std::string main_loop = "";

        do_emit_ = true;
        if (auto class_decl = dyn_cast<CXXRecordDecl>(find_decl_by_name(class_name)))
        {
            note_found(class_decl->getLocation(), "class");
            // emit("struct VolumeRenderer_slang_UBO_Data{\n");
            // for (auto i : class_decl->fields())
            // {
            //     std::string type_str = i->getType().getAsString({{}});
            //     if (type_str.find("vector") != std::string::npos)
            //     {
            //         emit_indent(1);
            //         emit("uint ");
            //         emit(i->getNameAsString());
            //         emit("_size;\n");
            //         emit_indent(1);
            //         emit("uint ");
            //         emit(i->getNameAsString());
            //         emit("_capacity;\n");
            //     }
            //     else
            //     {
            //         emit_indent(1);
            //         emit(type_str);
            //         emit(" ");
            //         emit(i->getNameAsString());
            //         emit(";\n");
            //     }
            //     // outs() << type_str << "\n";
            //     // i->print(outs());
            //     // outs() << "\n";
            // }
            // emit("   uint dummy_last;\n }\n");
            // exit(1);
            for (auto method_decl : class_decl->methods())
            {
                if (method_decl->getNameAsString() == kernel_name)
                {
                    method_decl = dyn_cast<CXXMethodDecl>(method_decl->getDefinition());
                    note_found(method_decl->getLocation(), "kernel");
                    emit(method_decl->getNameAsString());
                    emit(":\n");

                    for (auto i : method_decl->getBody()->children())
                    {
                        if (auto for_stmt = dyn_cast<ForStmt>(i))
                        {
                            note_found(for_stmt->getForLoc(), "root for loop");
                            buffer_ = "";
                            inside_kernel_ = true;
                            rewrite_statement(for_stmt->getBody(), 1);
                            inside_kernel_ = false;
                            main_loop = buffer_;
                        }
                    }
                    break;
                }
            }
        }

        for (size_t i = 0; i < called_methods.size(); i++)
        {
            outs() << "Method: " << called_methods[i]->getNameAsString() << "\n";
            rewrite_function(called_methods[i]);
        }

        for (size_t i = 0; i < called_functions.size(); i++)
        {
            outs() << "Function: " << called_functions[i]->getNameAsString() << "\n";
            rewrite_function(called_functions[i]);
        }

        // outs() << buffer_ << "\n";
        std::ofstream f("out.slang");

        f << kernel_args;
        f << ubo;
        f << uniforms;
        // f << buffer_;
        for (auto &[decl, code] : functions)
        {
            for (char c : code)
            {
                if (c == '{')
                {
                    f << ";\n";
                    break;
                }
                f << c;
            }
            f << "\n";
        }

        f << slang_operators << "\n";

        for (auto &[decl, code] : functions)
        {
            f << code << "\n";
        }

        f << "// Main loop is here\n";

        f << kernel_boiler << main_loop << "}\n\n";
        // f << main_loop << "\n";
    }
}
