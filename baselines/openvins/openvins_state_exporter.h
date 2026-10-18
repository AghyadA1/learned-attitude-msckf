#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>

namespace ov_msckf {
class VioManager;
}

class OpenVinsStateExporter {
public:
  explicit OpenVinsStateExporter(const std::filesystem::path &output);
  void export_camera_update(std::int64_t timestamp_ns, const std::shared_ptr<ov_msckf::VioManager> &manager);

private:
  std::ofstream state_;
  std::ofstream covariance_;
  std::ofstream clones_;
};
