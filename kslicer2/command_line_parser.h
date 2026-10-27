#pragma once
#include <filesystem>
#include <optional>
#include <vector>

namespace kslicer {

    class CommandLineParser {
    public:
        template <typename T>
        void add_option(std::string name, T& destination) {
            Arg arg{};
            arg.name = name;
            arg.store = [](void* destination, const std::filesystem::path& value) {
                CommandLineParser::store(*static_cast<T*>(destination), value);
            };
            arg.destination = &destination;
            arg.is_optional = false;
            args_.push_back(arg);
        }

        template <typename T>
        void add_option(std::string name, std::optional<T>& destination) {
            Arg arg{};
            arg.name = name;
            arg.store = [](void* destination, const std::filesystem::path& value) {
                CommandLineParser::store(static_cast<std::optional<T>*>(destination)->emplace(), value);
            };
            arg.destination = &destination;
            arg.is_optional = true;
            args_.push_back(arg);
        }

        void add_flag(std::string name, bool& destination) {
            Arg arg{};
            arg.name = name;
            arg.store = nullptr;
            arg.destination = &destination;
            arg.is_optional = true;
            args_.push_back(arg);
        }

        bool parse(int argc, char** agrv) const;

    private:
        static void store(std::filesystem::path& destination, const std::filesystem::path& value) {
            destination = value;
        }

        static void store(std::string& destination, const std::filesystem::path& value) {
            destination = value.u8string();
        }

        struct Arg {
            std::string name;
            void (*store)(void*, const std::filesystem::path&) = nullptr;
            void* destination = nullptr;
            bool is_optional = false;
        };

        std::vector<Arg> args_;
    };

} // namespace kslicer
