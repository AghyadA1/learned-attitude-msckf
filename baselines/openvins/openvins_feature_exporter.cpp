#include "openvins_feature_exporter.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "core/VioManager.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/State.h"

using ov_msckf::VioManager;

OpenVinsFeatureExporter::OpenVinsFeatureExporter(const std::filesystem::path &output, std::string sequence)
    : sequence_(std::move(sequence)) {
  observations_.open(output / "feature_observations.csv");
  completed_.open(output / "completed_feature_tracks.csv");
  triangulated_.open(output / "triangulated_features.csv");
  updates_.open(output / "visual_update_audit.csv");
  if (!observations_ || !completed_ || !triangulated_ || !updates_) {
    throw std::runtime_error("unable to open feature export files");
  }
  observations_ << "sequence,timestamp_ns,timestamp_s,camera_id,track_id,u,v,x_norm,y_norm,response,"
                   "tracking_status,grid_cell,clone_timestamp_ns\n";
  completed_ << "sequence,track_id,observation_count,track_duration_s,first_timestamp_ns,last_timestamp_ns,"
                "x_W,y_W,z_W,triangulation_success,positive_depth_checks,triangulation_condition,"
                "reprojection_mean_px,reprojection_median_px,reprojection_p95_px,end_reason,"
                "accepted_msckf_update,update_rejection_reason\n";
  triangulated_ << "sequence,track_id,x_W,y_W,z_W,triangulation_success,positive_depth_checks,"
                   "reprojection_mean_px,reprojection_median_px,reprojection_p95_px\n";
  updates_ << "sequence,timestamp_ns,candidate_features,accepted_features,rejected_features,"
              "innovation_chi2_sum,innovation_dof,update_type\n";
  observations_ << std::setprecision(17);
  completed_ << std::setprecision(17);
  triangulated_ << std::setprecision(17);
  updates_ << std::setprecision(17);
}

void OpenVinsFeatureExporter::export_camera_update(std::int64_t timestamp_ns, int image_width, int image_height,
                                                   const std::shared_ptr<VioManager> &manager) {
  const double timestamp_s = static_cast<double>(timestamp_ns) * 1e-9;
  const auto ids_by_camera = manager->get_last_feature_ids();
  const auto keypoints_by_camera = manager->get_last_feature_observations();
  const auto database = manager->get_feature_database();
  const auto params = manager->get_params();
  for (const auto &[camera_id, ids] : ids_by_camera) {
    const auto keypoint_it = keypoints_by_camera.find(camera_id);
    if (keypoint_it == keypoints_by_camera.end() || keypoint_it->second.size() != ids.size()) {
      throw std::runtime_error("tracker ID/keypoint size mismatch");
    }
    for (std::size_t index = 0; index < ids.size(); ++index) {
      ov_core::Feature feature;
      if (!database->get_feature_clone(ids[index], feature)) {
        continue;
      }
      const auto camera_times = feature.timestamps.find(camera_id);
      const auto camera_uvs = feature.uvs.find(camera_id);
      const auto camera_norm = feature.uvs_norm.find(camera_id);
      if (camera_times == feature.timestamps.end() || camera_uvs == feature.uvs.end() || camera_norm == feature.uvs_norm.end() ||
          camera_times->second.empty()) {
        continue;
      }
      const auto &uv = camera_uvs->second.back();
      const auto &norm = camera_norm->second.back();
      const auto &keypoint = keypoint_it->second[index];
      const int cell_x = std::clamp(static_cast<int>(keypoint.pt.x * params.grid_x / image_width), 0, params.grid_x - 1);
      const int cell_y = std::clamp(static_cast<int>(keypoint.pt.y * params.grid_y / image_height), 0, params.grid_y - 1);
      const auto clone_ns = manager->initialized() ? timestamp_ns : -1;
      observations_ << sequence_ << ',' << timestamp_ns << ',' << timestamp_s << ',' << camera_id << ',' << ids[index] << ',' << uv(0)
                    << ',' << uv(1) << ',' << norm(0) << ',' << norm(1) << ',' << keypoint.response << ",tracked," << cell_x << ':'
                    << cell_y << ',' << clone_ns << '\n';
    }
  }

  if (!manager->initialized()) {
    observations_.flush();
    return;
  }
  const auto diagnostics = manager->get_last_msckf_diagnostics();
  std::size_t new_candidates = 0;
  std::size_t new_accepted = 0;
  double innovation_chi2_sum = 0.0;
  int innovation_dof = 0;
  for (const auto &diagnostic : diagnostics) {
    if (!completed_ids_.insert(diagnostic.feature_id).second) {
      continue;
    }
    ++new_candidates;
    const bool is_accepted = diagnostic.accepted;
    new_accepted += is_accepted ? 1 : 0;
    if (diagnostic.innovation_chi2 >= 0) {
      innovation_chi2_sum += diagnostic.innovation_chi2;
      innovation_dof += diagnostic.innovation_dof;
    }
    const std::int64_t first_ns = std::isfinite(diagnostic.first_timestamp)
                                      ? static_cast<std::int64_t>(std::llround(diagnostic.first_timestamp * 1e9))
                                      : -1;
    const std::int64_t last_ns = std::isfinite(diagnostic.last_timestamp)
                                     ? static_cast<std::int64_t>(std::llround(diagnostic.last_timestamp * 1e9))
                                     : -1;
    const double duration = diagnostic.first_timestamp >= 0 && diagnostic.last_timestamp >= 0
                                ? diagnostic.last_timestamp - diagnostic.first_timestamp
                                : 0.0;
    const bool positive_depth =
        diagnostic.depth_check_count > 0 && diagnostic.positive_depth_count == diagnostic.depth_check_count;
    const char *end_reason = is_accepted ? "msckf_update" : "msckf_candidate_rejected";
    completed_ << sequence_ << ',' << diagnostic.feature_id << ',' << diagnostic.observation_count << ',' << duration << ','
               << first_ns << ',' << last_ns << ',' << diagnostic.position_global(0) << ',' << diagnostic.position_global(1) << ','
               << diagnostic.position_global(2) << ',' << (diagnostic.triangulation_success ? 1 : 0) << ','
               << (positive_depth ? 1 : 0) << ',' << diagnostic.triangulation_condition << ','
               << diagnostic.reprojection_mean_px << ',' << diagnostic.reprojection_median_px << ','
               << diagnostic.reprojection_p95_px << ',' << end_reason << ',' << (is_accepted ? 1 : 0) << ','
               << diagnostic.rejection_reason << '\n';
    if (diagnostic.triangulation_success) {
      triangulated_ << sequence_ << ',' << diagnostic.feature_id << ',' << diagnostic.position_global(0) << ','
                    << diagnostic.position_global(1) << ',' << diagnostic.position_global(2) << ",1,"
                    << (positive_depth ? 1 : 0) << ',' << diagnostic.reprojection_mean_px << ','
                    << diagnostic.reprojection_median_px << ',' << diagnostic.reprojection_p95_px << '\n';
    }
  }
  if (new_candidates > 0) {
    updates_ << sequence_ << ',' << timestamp_ns << ',' << new_candidates << ',' << new_accepted << ','
             << (new_candidates - new_accepted) << ',' << innovation_chi2_sum << ',' << innovation_dof << ",msckf\n";
  }
  observations_.flush();
  completed_.flush();
  triangulated_.flush();
  updates_.flush();
}
