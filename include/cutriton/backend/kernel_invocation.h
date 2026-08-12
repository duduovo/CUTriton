#pragma once

#include <string>
#include <variant>
#include <vector>

#include "cutriton/backend/kernel_artifact.h"

namespace cutriton {

struct KernelInvocation {
  KernelArtifact artifact;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

struct ViewInvocation {
  std::string input;
  std::string output;
};

struct CandidateTemporary {
  std::string name;
  std::string like_value;
  std::size_t slot{0};
};

using PlanStep = std::variant<KernelInvocation, ViewInvocation>;

struct ExecutionCandidate {
  std::string candidate_id;
  std::vector<PlanStep> steps;
  std::vector<CandidateTemporary> temporaries;
  bool is_default{false};
};

}  // namespace cutriton
