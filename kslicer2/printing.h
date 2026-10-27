#pragma once
#include <cstdio>

#define printf(...) DO_NOT_USE_PRINTF_USE_KSLICER_OUTS()
#define fprintf(...) DO_NOT_USE_PRINTF_USE_KSLICER_ERRS()

#define cout DO_NOT_USE_COUT_AND_DO_NOT_INCLUDE_IOSTREAM_USE_KSLICER_OUTS
#define cerr DO_NOT_USE_CERR_AND_DO_NOT_INCLUDE_IOSTREAM_USE_KSLICER_ERRS
#define clog DO_NOT_USE_CLOG_AND_DO_NOT_INCLUDE_IOSTREAM_USE_KSLICER_ERRS_
#define cin DO_NOT_USE_CIN_AND_DO_NOT_INCLUDE_IOSTREAM

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
