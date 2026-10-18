#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>

namespace ov_msckf {
class VioManager;
}

class OpenVinsFeatureExporter {
public:
  OpenVinsFeatureExporter(const std::filesystem::path &output, std::string sequence);
  void export_camera_update(std::int64_t timestamp_ns, int image_width, int image_height,
                            const std::shared_ptr<ov_msckf::VioManager> &manager);

private:
  std::string sequence_;
  std::ofstream observations_;
  std::ofstream completed_;
  std::ofstream triangulated_;
  std::ofstream updates_;
  std::unordered_set<std::size_t> completed_ids_;
};
