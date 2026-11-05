#include "feature_status.h"
#include "kslicer_containers.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>

namespace
{
  std::string Trim(const std::string& a_str)
  {
    const size_t first = a_str.find_first_not_of(" \t\r");
    if(first == std::string::npos)
      return "";
    const size_t last = a_str.find_last_not_of(" \t\r");
    return a_str.substr(first, last - first + 1);
  }

  // '| a | b |' ==> {"a", "b"}
  std::vector<std::string> SplitTableRow(const std::string& a_line)
  {
    std::vector<std::string> cells;
    std::string line = Trim(a_line);
    if(line.size() < 2 || line.front() != '|')
      return cells;
    line = line.substr(1);
    if(line.back() == '|')
      line.pop_back();
    std::stringstream ss(line);
    std::string cell;
    while(std::getline(ss, cell, '|'))
      cells.push_back(Trim(cell));
    return cells;
  }

  // name of the column in 'doc/features.md' for the value of '-shaderCC'; empty if the back end is not tracked
  std::string BackendColumn(const std::string& a_shaderCC)
  {
    std::string name = a_shaderCC;
    for(auto& c : name)
      c = char(std::tolower(c));
    if(name == "slang")
      return "Slang";
    if(name == "glsl")
      return "GLSL";
    if(name == "cuda" || name == "hip" || name == "musa")
      return "CUDA/HIP";
    if(name == "ispc")
      return "ISPC";
    return "";
  }

  struct FeatureRow
  {
    std::string status;  // the cell of the selected back end
    std::string name;    // the 'Feature' column
  };
}

bool kslicer::CheckRequiredFeatures(const std::vector<std::string>& a_features, const std::string& a_shaderCC, const std::string& a_featureFile)
{
  if(a_features.empty())
    return true;

  const std::string column = BackendColumn(a_shaderCC);
  if(column == "")
  {
    std::cout << "[kslicer]: error: 'requireFeatures' is given, but feature status of back end '" << a_shaderCC.c_str()
              << "' is not tracked in " << a_featureFile.c_str() << std::endl;
    return false;
  }

  std::ifstream fin(a_featureFile);
  if(!fin.is_open())
  {
    std::cout << "[kslicer]: error: 'requireFeatures' is given, but " << a_featureFile.c_str()
              << " is not found (run kslicer from the repository root or pass '-selfdir')" << std::endl;
    return false;
  }

  // every table has a header '| ID | Feature | Slang | GLSL | ...'; rows start with '| `id` |'
  kslicer::umap<std::string, FeatureRow> rows;
  int statusCol = -1;
  int nameCol   = -1;
  std::string line;
  while(std::getline(fin, line))
  {
    const auto cells = SplitTableRow(line);
    if(cells.empty())
    {
      statusCol = -1;   // end of the table
      continue;
    }
    if(cells[0] == "ID")
    {
      statusCol = -1;
      nameCol   = -1;
      for(size_t i = 0; i < cells.size(); i++)
      {
        if(cells[i] == column)
          statusCol = int(i);
        if(cells[i] == "Feature")
          nameCol = int(i);
      }
      continue;
    }
    const std::string& idCell = cells[0];
    if(statusCol < 0 || idCell.size() < 3 || idCell.front() != '`' || idCell.back() != '`' || int(cells.size()) <= statusCol)
      continue;
    FeatureRow row;
    row.status = cells[statusCol];
    row.name   = (nameCol >= 0 && nameCol < int(cells.size())) ? cells[nameCol] : "";
    rows[idCell.substr(1, idCell.size() - 2)] = row;
  }

  bool allSupported = true;
  for(const auto& id : a_features)
  {
    auto p = rows.find(id);
    if(p == rows.end())
    {
      std::cout << "[kslicer]: error: unknown feature '" << id.c_str() << "' in 'requireFeatures', see " << a_featureFile.c_str() << std::endl;
      allSupported = false;
      continue;
    }
    const std::string& st = p->second.status;
    const std::string  descr = "feature '" + id + "' (" + p->second.name + ")";
    if(st.find("✅") != std::string::npos)
      continue;
    else if(st.find("🧪") != std::string::npos)
      std::cout << "[kslicer]: WARNING, " << descr.c_str() << " is experimental in back end " << column.c_str() << std::endl;
    else if(st.find("🟡") != std::string::npos)
      std::cout << "[kslicer]: WARNING, " << descr.c_str() << " is supported partially by back end " << column.c_str()
                << ", see the notes in " << a_featureFile.c_str() << std::endl;
    else
    {
      const char* what = (st.find("❌") != std::string::npos) ? "is not supported" : "has unknown status";
      std::cout << "[kslicer]: error: " << descr.c_str() << " " << what << " in back end " << column.c_str()
                << ", see " << a_featureFile.c_str() << std::endl;
      allSupported = false;
    }
  }
  return allSupported;
}
