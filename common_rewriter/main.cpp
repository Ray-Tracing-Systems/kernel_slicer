#include "clang/Tooling/Tooling.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"
#include "common_rewriter.h"

using namespace clang;
using namespace clang::tooling;

// ASTConsumer that runs our visitor.
class MyASTConsumer : public ASTConsumer
{
public:
    void HandleTranslationUnit(ASTContext &context) override
    {
        common_rewriter::rewrite_kernel("VolumeRenderer", "kernel1D_RaymarchGrid_DDA");
    }
};

class MyASTFrontedAction : public ASTFrontendAction
{
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override
    {
        common_rewriter::compiler_instance = &CI;
        return std::make_unique<MyASTConsumer>();
    }
};

static llvm::cl::OptionCategory ToolCategory{"What is a tool category???"};

int main(int argc, const char **argv)
{
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, ToolCategory);
    if (!ExpectedParser)
    {
        llvm::errs() << ExpectedParser.takeError() << "\n";
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();

    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());

    return Tool.run(newFrontendActionFactory<MyASTFrontedAction>().get());
}
