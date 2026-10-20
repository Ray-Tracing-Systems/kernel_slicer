
#include "utils.h"
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/SemaDiagnostic.h>

namespace kslicer {
    using namespace clang;

    struct SizeAttrInfo : public ParsedAttrInfo {

        static constexpr Spelling spellings[] = {
            {ParsedAttr::AS_GNU, "size"},             // __attribute__((size("a_dataSize")))
            {ParsedAttr::AS_CXX11, "size"},           // [[size("a_dataSize")]]
            {ParsedAttr::AS_CXX11, "kslicer::size"}}; // [[kslicer::size("a_dataSize")]]

        SizeAttrInfo() {
            NumArgs = 1;
            Spellings = spellings;
        }

        bool diagAppertainsToDecl(Sema& sema, const ParsedAttr& attr, const Decl* decl) const override {
            if (!isa<ParmVarDecl>(decl)) {
                sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str) << attr << "function parameters";
                return false;
            }
            return true;
        }

        AttrHandling handleDeclAttribute(Sema& sema, Decl* decl, const ParsedAttr& attr) const override {

            KSLICER_ACCERT(attr.getNumArgs() == 1);
            Expr* args[] = {attr.getArgAsExpr(0)};
            decl->addAttr(AnnotateAttr::Create(sema.Context, "kslicer::size", args, 1, attr.getRange()));
            return AttributeApplied;
        }
    };

    static ParsedAttrInfoRegistry::Add<SizeAttrInfo>
        kslicer_size_attribute("kslicer::size", "Specifies that pointer declaration points to an array of given size");
} // namespace kslicer
