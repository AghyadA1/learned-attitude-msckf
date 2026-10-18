#include "openvins_state_exporter.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "core/VioManager.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "utils/quat_ops.h"

using ov_msckf::StateHelper;
using ov_msckf::VioManager;

namespace {

void require_finite(const Eigen::MatrixXd &value, const char *name) {
  if (!value.allFinite()) {
    throw std::runtime_error(std::string(name) + " contains NaN or infinity");
  }
}

} // namespace

OpenVinsStateExporter::OpenVinsStateExporter(const std::filesystem::path &output) {
  std::filesystem::create_directories(output);
  state_.open(output / "state_trajectory.csv");
  covariance_.open(output / "state_covariance_summary.csv");
  clones_.open(output / "clone_poses.csv");
  if (!state_ || !covariance_ || !clones_) {
    throw std::runtime_error("unable to open state export files");
  }
  state_ << "timestamp_ns,timestamp_s,qx,qy,qz,qw,p_x_m,p_y_m,p_z_m,v_x_m_s,v_y_m_s,v_z_m_s,"
            "bg_x_rad_s,bg_y_rad_s,bg_z_rad_s,bg_x_deg_s,bg_y_deg_s,bg_z_deg_s,"
            "ba_x_m_s2,ba_y_m_s2,ba_z_m_s2\n";
  covariance_ << "timestamp_ns,timestamp_s,covariance_dimension,symmetry_error_fro,minimum_eigenvalue,"
                 "maximum_eigenvalue,condition_estimate,cholesky_success,clone_count,update_type\n";
  clones_ << "update_timestamp_ns,clone_timestamp_ns,clone_timestamp_s,camera_id,qx,qy,qz,qw,"
             "p_x_m,p_y_m,p_z_m,covariance_index\n";
  state_ << std::setprecision(17);
  covariance_ << std::setprecision(17);
  clones_ << std::setprecision(17);
}

void OpenVinsStateExporter::export_camera_update(std::int64_t timestamp_ns, const std::shared_ptr<VioManager> &manager) {
  if (!manager->initialized()) {
    return;
  }
  const auto state = manager->get_state();
  const auto imu = state->_imu;
  const auto q = imu->quat();
  const auto p = imu->pos();
  const auto v = imu->vel();
  const auto bg = imu->bias_g();
  const auto ba = imu->bias_a();
  require_finite(q, "quaternion");
  require_finite(p, "position");
  require_finite(v, "velocity");
  require_finite(bg, "gyro bias");
  require_finite(ba, "accel bias");
  constexpr double rad_to_deg = 180.0 / 3.14159265358979323846;
  state_ << timestamp_ns << ',' << state->_timestamp << ',' << q(0) << ',' << q(1) << ',' << q(2) << ',' << q(3) << ','
         << p(0) << ',' << p(1) << ',' << p(2) << ',' << v(0) << ',' << v(1) << ',' << v(2) << ',' << bg(0) << ','
         << bg(1) << ',' << bg(2) << ',' << bg(0) * rad_to_deg << ',' << bg(1) * rad_to_deg << ',' << bg(2) * rad_to_deg
         << ',' << ba(0) << ',' << ba(1) << ',' << ba(2) << '\n';

  const Eigen::MatrixXd covariance = StateHelper::get_full_covariance(state);
  require_finite(covariance, "covariance");
  const double symmetry_error = (covariance - covariance.transpose()).norm();
  const Eigen::MatrixXd diagnostic_copy = 0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(diagnostic_copy, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("covariance eigensolver failed");
  }
  const double minimum = solver.eigenvalues().minCoeff();
  const double maximum = solver.eigenvalues().maxCoeff();
  double smallest_nonzero = std::numeric_limits<double>::infinity();
  for (const double eigenvalue : solver.eigenvalues()) {
    const double magnitude = std::abs(eigenvalue);
    if (magnitude > std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(maximum))) {
      smallest_nonzero = std::min(smallest_nonzero, magnitude);
    }
  }
  const double condition = std::isfinite(smallest_nonzero) ? std::abs(maximum) / smallest_nonzero : 1.0;
  Eigen::LLT<Eigen::MatrixXd> cholesky(diagnostic_copy);
  covariance_ << timestamp_ns << ',' << state->_timestamp << ',' << covariance.rows() << ',' << symmetry_error << ',' << minimum << ','
              << maximum << ',' << condition << ',' << (cholesky.info() == Eigen::Success ? 1 : 0) << ','
              << state->_clones_IMU.size() << ",camera\n";
  if (minimum < -1e-9) {
    throw std::runtime_error("materially negative covariance eigenvalue: " + std::to_string(minimum));
  }
  if (!std::isfinite(condition) || condition > 1e20) {
    throw std::runtime_error("unbounded covariance condition estimate: " + std::to_string(condition));
  }
  if (covariance.rows() < 15 + 6 * static_cast<Eigen::Index>(state->_clones_IMU.size())) {
    throw std::runtime_error("covariance dimension is inconsistent with clone count");
  }

  const auto calibration = state->_calib_IMUtoCAM.at(0);
  for (const auto &[clone_time, clone] : state->_clones_IMU) {
    const Eigen::Matrix3d r_g_to_c = calibration->Rot() * clone->Rot();
    const Eigen::Vector3d p_c_in_g = clone->pos() - r_g_to_c.transpose() * calibration->pos();
    const Eigen::Vector4d q_g_to_c = ov_core::rot_2_quat(r_g_to_c);
    require_finite(q_g_to_c, "clone camera quaternion");
    require_finite(p_c_in_g, "clone camera position");
    const auto clone_ns = static_cast<std::int64_t>(std::llround(clone_time * 1e9));
    clones_ << timestamp_ns << ',' << clone_ns << ',' << clone_time << ",0," << q_g_to_c(0) << ',' << q_g_to_c(1) << ','
            << q_g_to_c(2) << ',' << q_g_to_c(3) << ',' << p_c_in_g(0) << ',' << p_c_in_g(1) << ',' << p_c_in_g(2) << ','
            << clone->id() << '\n';
  }
  state_.flush();
  covariance_.flush();
  clones_.flush();
}
