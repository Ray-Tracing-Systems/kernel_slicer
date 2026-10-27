#include "template_rendering.h"
#include "io.h"
#include <inja/inja.hpp>
#include <llvm/Support/WithColor.h>

namespace kslicer {

    bool render_template(nlohmann::json data, std::filesystem::path template_path, std::filesystem::path output_path) {
        std::string result;
        try {
            inja::Environment env;
            env.set_trim_blocks(true);
            env.set_lstrip_blocks(true);
            inja::Template t = env.parse_template(template_path);
            result = env.render(t, data);
        } catch (const inja::FileError& e) {
            // If inja failed to open std::ifstream then we try to open file ourselves.
            // If it is really an error then read_entire_text_file() will write error message into errs()
            // If it is it succeeded (which is strange) then we just print what ever was inja's error message
            if (read_entire_text_file(template_path, result)) {
                kslicer_error() << e.message << "\n";
            }
            return false;
        } catch (const inja::InjaError& e) {
            llvm::WithColor::error(errs(), template_path.u8string() + ":" + std::to_string(e.location.line) + ":" +
                                               std::to_string(e.location.column))
                << e.message << "\n";
            return false;
        }

        return write_entire_text_file(output_path, result);
    }

} // namespace kslicer
