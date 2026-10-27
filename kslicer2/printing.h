#pragma once
#include <filesystem>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>

namespace kslicer {
    using llvm::outs, llvm::errs;

    // prints kslicer: error:
    inline llvm::raw_ostream& kslicer_error() { return llvm::WithColor::error(llvm::errs(), "kslicer"); }

    // prints kslicer: warning:
    llvm::raw_ostream& kslicer_warning();

    inline llvm::raw_ostream& operator<<(llvm::raw_ostream& s, const std::filesystem::path& p) {
        return s << p.u8string();
    }

} // namespace kslicer
