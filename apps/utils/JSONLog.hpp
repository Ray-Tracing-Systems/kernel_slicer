#include <iostream>
#include <ostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>


std::string quoted(const std::string &str);

template <typename T>
std::string serialize(const T &object) {
  return std::to_string(object);
}

std::string serialize(const std::string &object);

std::string serialize(const char *object);

std::string serialize(const bool &object);

template <typename T>
std::string serialize(const std::vector<T> &vec) {
  std::string result = "[";
  size_t size = vec.size();
  for (size_t i = 0; i < size; ++i) {
    result += serialize(vec[i]) + ((i == size-1) ? "]" : ", ");
  }
  return result;
}

template <typename T, std::size_t N>
std::string serialize(const std::array<T, N> &arr) {
  std::string result = "[";
  for (size_t i = 0; i < N; ++i) {
    result += serialize(arr[i]) + ((i == N-1) ? "]" : ", ");
  }
  return result;
}

template <typename T, std::size_t N>
std::string serialize(T(&arr)[N]) {
  std::string result = "[";
  for (size_t i = 0; i < N; ++i) {
    result += serialize(arr[i]) + ((i == N-1) ? "]" : ", ");
  }
  return result;
}


class JSONLog {
public:
  JSONLog() = delete;

  template <typename T>
  static void write(const std::string &name, T data) {
    std::string s_data = serialize(data);
    std::cout << name << " : " << s_data << std::endl;
    _json.values[name] = s_data;
  }
  template <typename T, std::size_t N>
  static void write(const std::string &name, T(&arr)[N]) {
    std::string s_data = serialize(arr);
    std::cout << name << " : " << s_data << std::endl;
    _json.values[name] = s_data;
  }

  static void saveToFile(const std::string &filename) {
    std::ofstream file(filename);
    if (file) {
      file << "{" << std::endl;
      int i = 0;
      int size = _json.values.size();
      for (const auto& kv : _json.values) {
        file << "\t"+quoted(kv.first)+" : "+kv.second;
        if (i < size-1) {
          file << ",";
        }
        file << std::endl;
      }
      file << "}" << std::endl;
    } else {
      std::stringstream ss;
      ss << "Can't save JSON: can't open file: " << filename;
      throw std::invalid_argument(ss.str());
    }
    _json.clear();
  }

private:
  
  class DataHolder {
  public:
    std::unordered_map<std::string, std::string> values;

    void clear() {
      values = {};
    }
    ~DataHolder() noexcept(false) {
      if (!values.empty()) {
        throw std::runtime_error("JSONLog isn't saved to file");
      }
    }
  };
  static DataHolder _json;
};

#ifdef JSON_LOG_IMPLEMENTATION
JSONLog::DataHolder JSONLog::_json;


std::string quoted(const std::string &str) {
  return "\""+str+"\"";
}

std::string serialize(const std::string &object) {
  return quoted(object);
}

std::string serialize(const char *object) {
  return quoted(object);
}

std::string serialize(const bool &object) {
  return object ? "true" : "false";
}
#endif