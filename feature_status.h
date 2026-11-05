#pragma once
#include <string>
#include <vector>

namespace kslicer
{
  /**
  \brief Checks features required by the config ("requireFeatures" in kmake.json) against the status table in 'doc/features.md'.
  \param a_features    -- feature IDs from the config
  \param a_shaderCC    -- value of '-shaderCC'
  \param a_featureFile -- path to 'doc/features.md'
  \return false if some feature is not supported (or its status is unknown) by the selected back end and translation must stop

   Statuses: supported and experimental features are accepted (experimental and partial ones with a warning),
   not supported and unknown ones are errors. An unknown feature ID is an error too.
  */
  bool CheckRequiredFeatures(const std::vector<std::string>& a_features, const std::string& a_shaderCC, const std::string& a_featureFile);
};
