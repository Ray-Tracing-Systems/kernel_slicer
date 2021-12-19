#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <optional>


class ArgParser {
public:
  ArgParser (int &argc, const char **argv) {
    for (int i=1; i < argc; ++i) {
      _tokens.push_back(std::string(argv[i]));
    }
  }

  template <typename T>
  T getOptionValue(const std::string &option, std::optional<T> defaultValue=std::nullopt) const {
    std::vector<std::string>::const_iterator itr;
    
    itr = std::find(_tokens.begin(), _tokens.end(), option);
    if (itr == _tokens.end() || ++itr == _tokens.end()) {
      if (defaultValue.has_value()) {
        return defaultValue.value();
      }
      std::stringstream ss;
      ss << "Option "<<option<<" doesn't have any value";
      throw std::runtime_error(ss.str());
    }

    return from_string<T>(*itr);
  }

  bool hasOption(const std::string &option) const {
    return std::find(_tokens.begin(), _tokens.end(), option) != _tokens.end();
  }

private:
  std::vector<std::string> _tokens;

  template <typename T>
  static T from_string(const std::string &str);
};


template <>
std::string ArgParser::from_string(const std::string &str) {
	return str;
}

template <>
int ArgParser::from_string(const std::string &str) {
	return std::stoi(str);
}

template <>
float ArgParser::from_string(const std::string &str) {
	return std::stof(str);
}