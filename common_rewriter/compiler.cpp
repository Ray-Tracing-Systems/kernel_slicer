#include "compiler.h"
#include "utils.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>

namespace kslicer {
    using namespace llvm;
    using namespace clang;
    using namespace clang::tooling;

    static CompilerInstance* compiler_ = nullptr;
    CompilerInstance* compiler() {
        REWRITE_ASSERT(compiler_);
        return compiler_;
    }

    class MyASTConsumer : public ASTConsumer {
    public:
        MyASTConsumer(std::function<void()> callback) : callback_(callback) {}
        void HandleTranslationUnit(ASTContext&) override { callback_(); }

    private:
        std::function<void()> callback_;
    };

    class MyASTFrontedAction : public ASTFrontendAction {
    public:
        MyASTFrontedAction(std::function<void()> callback) : callback_(callback) {}
        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& compiler, StringRef) override {
            compiler_ = &compiler;
            return std::make_unique<MyASTConsumer>(callback_);
        }

    private:
        std::function<void()> callback_;
    };

    class MyFrontendActionFactory : public FrontendActionFactory {
    public:
        MyFrontendActionFactory(std::function<void()> callback) : callback_(callback) {}
        std::unique_ptr<clang::FrontendAction> create() override {
            return std::make_unique<MyASTFrontedAction>(callback_);
        }

    private:
        std::function<void()> callback_;
    };

    int invoke_compiler(const std::vector<std::string>& sources, const std::vector<std::string>& params,
                        std::function<void()> callback) {
        FixedCompilationDatabase db(".", params);
        ClangTool tool(db, sources);
        MyFrontendActionFactory action_factory(callback);
        int res = tool.run(&action_factory);
        compiler_ = nullptr;
        return res;
    }

    int invoke_compiler(const InvocationParams& params, std::function<void()> callback) {

        std::vector<std::string> sources;
        std::vector<std::string> args;

        for (auto i : params.sources) {
            sources.push_back(i.string());
        }

        for (auto i : params.include_directories) {
            args.push_back("-I");
            args.push_back(i.string());
        }

        return invoke_compiler(sources, args, callback);
    }

} // namespace kslicer
