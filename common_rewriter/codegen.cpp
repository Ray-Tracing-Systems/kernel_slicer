#include "codegen.h"
#include "errors.h"
#include "utils.h"

namespace common_rewriter {
    using llvm::outs;

    void Codegen::emit_indent(size_t indent) {
        for (size_t i = 0; i < indent; i++) {
            emit("    ");
        }
    }

    void Codegen::emit(std::string str) { buffer += str; }

    void Codegen::emit(llvm::StringRef str) { emit(std::string(str.begin(), str.end())); }

    void Codegen::emit(const char* str) { emit(std::string(str)); }

    void Codegen::render_type(QualType type) { emit(render_type_name(type)); }

    // std::string Codegen::resolve_uniform_member(std::string name) { return name; }

    // std::string Codegen::resolve_function_param(std::string name) { return name; }

    // std::string Codegen::resolve_buffer_member(std::string name) { return name; }

    void Codegen::render_function_param(const ParmVarDecl* param) { emit(param->getNameAsString()); }

    void Codegen::render_method_call(const CXXMemberCallExpr* expr) {
        if (!dyn_cast<CXXThisExpr>(expr->getImplicitObjectArgument())) {
            render_expression(expr->getImplicitObjectArgument());
            emit(".");
        }
        emit(expr->getMethodDecl()->getNameAsString());
        emit("(");
        for (int i = 0; i < expr->getNumArgs(); i++) {
            render_expression(expr->getArg(i));
            if (i < expr->getNumArgs() - 1)
                emit(", ");
        }
        emit(")");
    }

    void Codegen::render_field_access(const MemberExpr* expr) {
        if (dyn_cast<CXXThisExpr>(expr->getBase())) {
            emit(expr->getMemberDecl()->getNameAsString());
        } else {
            render_expression(expr->getBase());
            std::string field_name = expr->getMemberDecl()->getNameAsString();
            if (field_name.length() > 0) {
                emit(".");
                emit(field_name);
            }
        }
    }

    void Codegen::render_statement(const Stmt* stmt_, size_t indent, bool newline) {
        if (auto stmt = dyn_cast<CompoundStmt>(stmt_)) {
            emit_indent(indent);
            emit("{\n");
            for (auto i : stmt->children()) {
                render_statement(i, indent + 1);
            }
            emit_indent(indent);
            emit("}\n");
        } else if (auto stmt = dyn_cast<Expr>(stmt_)) {
            emit_indent(indent);
            render_expression(stmt);
            emit(";");
            if (newline)
                emit("\n");
        } else if (auto stmt = dyn_cast<DeclStmt>(stmt_)) {

            for (auto i : stmt->getDeclGroup()) {
                auto var = dyn_cast<VarDecl>(i);
                REWRITE_ASSERT(var);
                emit_indent(indent);
                render_type(var->getType());
                emit(" ");
                emit(var->getNameAsString());
                if (var->getInit()) {
                    emit(" = ");
                    render_expression(var->getInit());
                }
                emit(";");
                if (newline)
                    emit("\n");
            }
        } else if (auto stmt = dyn_cast<IfStmt>(stmt_)) {
            emit_indent(indent);
            emit("if (");
            render_expression(stmt->getCond());
            emit(")\n");
            render_statement(stmt->getThen(), indent);
            if (stmt->getElse()) {
                emit_indent(indent);
                emit("else\n");
                render_statement(stmt->getElse(), indent);
            }
        } else if (auto stmt = dyn_cast<WhileStmt>(stmt_)) {
            emit_indent(indent);
            emit("while (");
            render_expression(stmt->getCond());
            emit(")\n");
            render_statement(stmt->getBody(), indent);
        } else if (auto stmt = dyn_cast<ForStmt>(stmt_)) {
            emit_indent(indent);
            emit("for (");
            render_statement(stmt->getInit(), 0, false);
            render_expression(stmt->getCond());
            emit("; ");
            render_expression(stmt->getInc());
            emit(")\n");
            render_statement(stmt->getBody(), indent + 1);
        } else if (auto stmt = dyn_cast<BreakStmt>(stmt_)) {
            emit_indent(indent);
            emit("break;\n");
        } else if (auto stmt = dyn_cast<ContinueStmt>(stmt_)) {
            emit_indent(indent);
            emit("continue;\n");
        } else if (auto stmt = dyn_cast<ReturnStmt>(stmt_)) {
            emit_indent(indent);
            emit("return ");
            if (stmt->getRetValue())
                render_expression(stmt->getRetValue());
            emit(";\n");
        } else if (auto stmt = dyn_cast<SwitchStmt>(stmt_)) {
            emit_indent(indent);
            emit("switch(");
            render_expression(stmt->getCond());
            emit(")\n");
            render_statement(stmt->getBody(), indent);
        } else if (auto stmt = dyn_cast<CaseStmt>(stmt_)) {
            emit_indent(indent);
            emit("case ");
            render_expression(stmt->getLHS());
            emit(":\n");
            render_statement(stmt->getSubStmt(), indent);
        } else if (auto stmt = dyn_cast<DefaultStmt>(stmt_)) {
            emit_indent(indent);
            emit("default:\n");
            render_statement(stmt->getSubStmt(), indent);
        } else {
            error_unknown_stmt_class(stmt_);
        }
    }

    void Codegen::render_expression(const Expr* expr_) {
        if (auto expr = dyn_cast<ExprWithCleanups>(expr_)) {
            // Lets not do destroying of temporary variables
            if (expr->getNumObjects() > 0)
                error_expr_needs_cleanups(expr);
            render_expression(expr->getSubExpr());
        } else if (auto expr = dyn_cast<FullExpr>(expr_)) {
            render_expression(expr->getSubExpr());
        } else if (auto expr = dyn_cast<ImplicitCastExpr>(expr_)) {
            // Ignoring implicit casts
            return render_expression(expr->getSubExpr());
        } else if (auto expr = dyn_cast<MaterializeTemporaryExpr>(expr_)) {
            // I don't know what it is...
            return render_expression(expr->getSubExpr());
        } else if (auto expr = dyn_cast<ParenExpr>(expr_)) {
            emit("(");
            render_expression(expr->getSubExpr());
            emit(")");
        } else if (auto expr = dyn_cast<CStyleCastExpr>(expr_)) {
            emit("(");
            render_type(expr->getType());
            emit(")");
            render_expression(expr->getSubExpr());
        } else if (auto expr = dyn_cast<CXXFunctionalCastExpr>(expr_)) {
            render_type(expr->getType());
            emit("(");
            render_expression(expr->getSubExpr());
            emit(")");
        } else if (auto expr = dyn_cast<CXXConstructExpr>(expr_)) {
            if (expr->getNumArgs() == 0) {
                std::string s = render_type_name(expr->getType());
                if (s == "float2")
                    emit("float2(0, 0)");
                else if (s == "float3")
                    emit("float3(0, 0, 0)");
                else if (s == "float4")
                    emit("float4(0, 0, 0, 0)");
                else
                    emit("???");
            } else {
                render_type(expr->getType());
                emit("(");
                for (size_t i = 0; i < expr->getNumArgs(); i++) {
                    render_expression(expr->getArg(i));
                    if (i < expr->getNumArgs() - 1)
                        emit(", ");
                }
                emit(")");
            }
        } else if (auto expr = dyn_cast<BinaryOperator>(expr_)) {
            render_expression(expr->getLHS());
            emit(" ");
            emit(BinaryOperator::getOpcodeStr(expr->getOpcode()));
            emit(" ");
            render_expression(expr->getRHS());
        } else if (auto expr = dyn_cast<UnaryOperator>(expr_)) {
            emit(UnaryOperator::getOpcodeStr(expr->getOpcode()));
            render_expression(expr->getSubExpr());
        } else if (auto expr = dyn_cast<CXXOperatorCallExpr>(expr_)) {
            // note_found(expr->getExprLoc(), "here");
            if (expr->getNumArgs() == 1) {
                std::string s = getOperatorSpelling(expr->getOperator());
                if (s == "+" || s == "-") {
                    emit(s);
                    render_expression(expr->getArg(0));
                } else {
                    error_unknown_overloaded_operator(expr);
                }
            } else {
                REWRITE_ASSERT(expr->getNumArgs() == 2);
                std::string s = getOperatorSpelling(expr->getOperator());

                std::string known_binary_operators[] = {
                    "=", "+", "-", "*", "/", "+=", "-=", "*=", "/=", "&", "|", "<<", ">>"};

                if (std::find(std::begin(known_binary_operators), std::end(known_binary_operators), s) !=
                    std::end(known_binary_operators)) {
                    render_expression(expr->getArg(0));
                    emit(" ");
                    emit(s);
                    emit(" ");
                    render_expression(expr->getArg(1));
                } else if (s == "[]") {
                    render_expression(expr->getArg(0));
                    emit("[");
                    render_expression(expr->getArg(1));
                    emit("]");
                } else {
                    error_unknown_overloaded_operator(expr);
                }
            }
        } else if (auto expr = dyn_cast<ArraySubscriptExpr>(expr_)) {
            render_expression(expr->getBase());
            emit("[");
            render_expression(expr->getIdx());
            emit("]");
        } else if (auto expr = dyn_cast<ConditionalOperator>(expr_)) {
            render_expression(expr->getCond());
            emit(" ? ");
            render_expression(expr->getTrueExpr());
            emit(" : ");
            render_expression(expr->getFalseExpr());
        } else if (auto expr = dyn_cast<IntegerLiteral>(expr_)) {
            emit(string_from_source_range(expr->getSourceRange()));
        } else if (auto expr = dyn_cast<FloatingLiteral>(expr_)) {
            emit(string_from_source_range(expr->getSourceRange()));
        } else if (auto expr = dyn_cast<CXXBoolLiteralExpr>(expr_)) {
            emit(string_from_source_range(expr->getSourceRange()));
        } else if (auto expr = dyn_cast<DeclRefExpr>(expr_)) {
            auto decl = expr->getDecl();
            // if (decl->getDeclContext() == compiler_instance->getASTContext().getTranslationUnitDecl()) {
            //     bool found = false;
            //     for (auto i : globals)
            //         if (i == decl)
            //             found = true;
            //     if (!found)
            //         globals.push_back(dyn_cast<VarDecl>(decl));
            // }
            if (auto param = dyn_cast<ParmVarDecl>(decl))
                render_function_param(param);
            else
                emit(decl->getNameAsString());
        } else if (auto expr = dyn_cast<MemberExpr>(expr_)) {
            render_field_access(expr);
            // if (dyn_cast<CXXThisExpr>(expr->getBase())) {
            //     if (expr->getMemberDecl()->getType().getAsString({{}}).find("vector") != std::string::npos)
            //         emit(resolve_buffer_member(expr->getMemberDecl()->getNameAsString()));
            //     else
            //         emit(resolve_uniform_member(expr->getMemberDecl()->getNameAsString()));
            // } else {
            //     render_expression(expr->getBase());
            //     // Accessing anonimous unions inside structs
            //     if (expr->getMemberDecl()->getNameAsString().length() > 0) {
            //         emit(".");
            //         emit(expr->getMemberDecl()->getNameAsString());
            //     }
            // }
        } else if (auto expr = dyn_cast<CXXMemberCallExpr>(expr_)) {
            render_method_call(expr);
            // if (context != Context::CONTROL_FUNCTION && expr->getMethodDecl()->getNameAsString() == "size") {
            //     render_expression(expr->getImplicitObjectArgument());
            //     emit(resolve_uniform_member(
            //         dyn_cast<MemberExpr>(expr->getImplicitObjectArgument())->getMemberDecl()->getNameAsString()));
            //     emit("_size");
            // } else {
            //     if (!dyn_cast<CXXThisExpr>(expr->getImplicitObjectArgument())) {
            //         if (context == Context::CONTROL_FUNCTION) {
            //             render_expression(expr->getImplicitObjectArgument());
            //             emit(".");
            //         } else {
            //             error_can_only_call_methods_of_this(expr);
            //         }
            //     }

            //     if (context == Context::CONTROL_FUNCTION && dyn_cast<CXXThisExpr>(expr->getImplicitObjectArgument()))
            //     {

            //         emit("CALL_KERNEL(");
            //         emit(expr->getMethodDecl()->getNameAsString());
            //         emit(")");
            //     } else {

            //         emit(expr->getMethodDecl()->getNameAsString());
            //         emit("(");
            //         for (int i = 0; i < expr->getNumArgs(); i++) {
            //             render_expression(expr->getArg(i));
            //             if (i < expr->getNumArgs() - 1)
            //                 emit(", ");
            //         }
            //         emit(")");
            //     }
            // }
        } else if (auto expr = dyn_cast<CallExpr>(expr_)) {
            auto func = dyn_cast<FunctionDecl>(expr->getCalleeDecl());
            REWRITE_ASSERT(func);

            emit(func->getNameAsString());
            emit("(");
            for (int i = 0; i < expr->getNumArgs(); i++) {
                render_expression(expr->getArg(i));
                if (i < expr->getNumArgs() - 1)
                    emit(", ");
            }
            emit(")");
        } else {
            error_unknown_stmt_class(expr_);
        }
    }

    void ControlFunctionCodegen::render_method_call(const CXXMemberCallExpr* expr) {
        const ControlFunctionInfo* control_function = info_.get_control_function(decl_);
        bool is_kernel_call = false;

        for (auto i : control_function->kernel_calls) {
            if (i == expr) {
                is_kernel_call = true;
                const KernelInfo* kernel = info_.get_kernel(expr->getMethodDecl());
                size_t ds_index = ds_index_++;

                /*

                 {
                    vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                RaymarchGrid_BresenhamLayout, 0, 1, &m_allGeneratedDS[5], 0, nullptr);
                    RaymarchGrid_BresenhamCmd(m_packedXY_width * m_packedXY_height, sample %m_preset.spp,
                m_colorBuffer.data()); vkCmdPipelineBarrier(m_currCmdBuffer, prevStageBits,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr); prevStageBits =
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                }
                */
                emit("{ ");
                emit("vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ");
                emit(kernel->decl->getNameAsString().substr(9) + "Layout");
                emit(", 0, 1, &m_allGeneratedDS[");
                emit(std::to_string(ds_index));
                emit("], 0, nullptr);");

                emit(kernel->decl->getNameAsString().substr(9) + "Cmd");
                emit("(");
                for (int i = 0; i < expr->getNumArgs(); i++) {
                    render_expression(expr->getArg(i));
                    if (i < expr->getNumArgs() - 1)
                        emit(", ");
                }
                emit(")");
                emit(";");
                emit("vkCmdPipelineBarrier(m_currCmdBuffer, prevStageBits,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "
                     "0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);");
                emit("prevStageBits = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;");
                emit("}");

                break;
            }
        }
        if (!is_kernel_call) {
            Codegen::render_method_call(expr);
        }
    }

} // namespace common_rewriter
