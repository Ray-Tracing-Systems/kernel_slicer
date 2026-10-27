#pragma once

#include "ankerl/unordered_dense.h"

namespace kslicer
{
  // Hash containers with deterministic iteration order: elements are stored in order of insertion,
  // so generated code does not depend on hash values, bucket count or standard library implementation.
  // Erase moves the last element to the place of erased one (order is still deterministic).
  // Segmented storage keeps references and pointers to elements valid when new elements are inserted.
  //
  template<class Key, class T, class Hash = ankerl::unordered_dense::hash<Key>, class KeyEqual = std::equal_to<Key> >
  using umap = ankerl::unordered_dense::segmented_map<Key, T, Hash, KeyEqual>;

  template<class Key, class Hash = ankerl::unordered_dense::hash<Key>, class KeyEqual = std::equal_to<Key> >
  using uset = ankerl::unordered_dense::segmented_set<Key, Hash, KeyEqual>;
};
