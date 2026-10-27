#include "io.h"
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/WithColor.h>

namespace kslicer {

    llvm::raw_ostream& operator<<(llvm::raw_ostream& s, const std::filesystem::path& p) { return s << p.u8string(); }

    llvm::raw_ostream& kslicer_error() { return llvm::WithColor::error(llvm::errs(), "kslicer"); }

    // prints kslicer: warning:
    llvm::raw_ostream& kslicer_warning() { return llvm::WithColor::warning(llvm::errs(), "kslicer"); }

    bool read_entire_text_file(const std::filesystem::path& file_path, std::string& data) {
        if (auto buffer = llvm::MemoryBuffer::getFile(file_path.u8string(), true)) {
            data = buffer.get()->getBuffer().str();
            return true;
        } else {
            std::error_code e = buffer.getError();
            kslicer_error() << file_path.u8string() << ": " << e.message() << "\n";
            return false;
        }
    }

    bool write_entire_text_file(const std::filesystem::path& file_path, const std::string& data) {

        std::error_code e;
        llvm::raw_fd_ostream file(file_path.u8string(), e, llvm::sys::fs::OF_TextWithCRLF);
        if (e) {
            kslicer_error() << file_path.u8string() << ": " << e.message() << "\n";
            return false;
        }
        file << data;
        file.close();
        return true;
    }

} // namespace kslicer
