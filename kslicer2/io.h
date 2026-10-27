#pragma once
#include <filesystem>
#include <llvm/Support/raw_ostream.h>

namespace kslicer {
    using llvm::outs, llvm::errs;

    llvm::raw_ostream& operator<<(llvm::raw_ostream& s, const std::filesystem::path& p);

    // prints kslicer: error:
    llvm::raw_ostream& kslicer_error();

    // prints kslicer: warning:
    llvm::raw_ostream& kslicer_warning();

    bool read_entire_text_file(const std::filesystem::path& file_path, std::string& data);
    bool write_entire_text_file(const std::filesystem::path& file_path, const std::string& data);
} // namespace kslicer
