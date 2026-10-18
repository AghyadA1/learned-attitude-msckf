#include <Eigen/Core>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "openvins_feature_exporter.h"
#include "openvins_state_exporter.h"
#include "state/State.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace fs = std::filesystem;

struct ImuRow {
  std::int64_t timestamp_ns;
  Eigen::Vector3d gyro;
  Eigen::Vector3d accel;
};

struct CameraRow {
  std::int64_t timestamp_ns;
  fs::path image;
};

struct Arguments {
  fs::path mav0;
  fs::path config;
  fs::path output;
  std::size_t max_images = std::numeric_limits<std::size_t>::max();
};

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> values;
  std::stringstream stream(line);
  std::string value;
  while (std::getline(stream, value, ',')) {
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' || value.back() == '\t')) {
      value.pop_back();
    }
    const auto first = value.find_first_not_of(" \t");
    if (first != std::string::npos) {
      value.erase(0, first);
    }
    values.push_back(value);
  }
  return values;
}

std::int64_t parse_timestamp_ns(const std::string &text) {
  std::size_t parsed = 0;
  const auto timestamp = std::stoll(text, &parsed, 10);
  if (parsed != text.size() || timestamp < 0) {
    throw std::runtime_error("invalid integer nanosecond timestamp: " + text);
  }
  return timestamp;
}

template <typename Row> void validate_timestamps(const std::vector<Row> &rows, const std::string &name) {
  if (rows.empty()) {
    throw std::runtime_error(name + " stream is empty");
  }
  for (std::size_t index = 1; index < rows.size(); ++index) {
    if (rows[index].timestamp_ns <= rows[index - 1].timestamp_ns) {
      throw std::runtime_error(name + " timestamps are duplicate or non-monotonic");
    }
  }
}

std::vector<ImuRow> read_imu(const fs::path &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("unable to open " + path.string());
  }
  std::vector<ImuRow> rows;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto values = split_csv(line);
    if (values.size() != 7) {
      throw std::runtime_error("malformed IMU CSV row");
    }
    ImuRow row;
    row.timestamp_ns = parse_timestamp_ns(values[0]);
    row.gyro << std::stod(values[1]), std::stod(values[2]), std::stod(values[3]);
    row.accel << std::stod(values[4]), std::stod(values[5]), std::stod(values[6]);
    if (!row.gyro.allFinite() || !row.accel.allFinite()) {
      throw std::runtime_error("non-finite IMU measurement");
    }
    rows.push_back(row);
  }
  validate_timestamps(rows, "IMU");
  return rows;
}

std::vector<CameraRow> read_camera(const fs::path &csv_path, const fs::path &image_directory) {
  std::ifstream stream(csv_path);
  if (!stream) {
    throw std::runtime_error("unable to open " + csv_path.string());
  }
  std::vector<CameraRow> rows;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const auto values = split_csv(line);
    if (values.size() != 2) {
      throw std::runtime_error("malformed camera CSV row");
    }
    CameraRow row{parse_timestamp_ns(values[0]), image_directory / values[1]};
    if (!fs::is_regular_file(row.image)) {
      throw std::runtime_error("missing camera image: " + row.image.string());
    }
    rows.push_back(std::move(row));
  }
  validate_timestamps(rows, "camera");
  return rows;
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--mav0" && index + 1 < argc) {
      result.mav0 = argv[++index];
    } else if (option == "--config" && index + 1 < argc) {
      result.config = argv[++index];
    } else if (option == "--output" && index + 1 < argc) {
      result.output = argv[++index];
    } else if (option == "--max-images" && index + 1 < argc) {
      result.max_images = std::stoull(argv[++index]);
    } else {
      throw std::runtime_error("unknown or incomplete option: " + option);
    }
  }
  if (result.mav0.empty() || result.config.empty() || result.output.empty()) {
    throw std::runtime_error("usage: euroc_file_runner --mav0 PATH --config YAML --output PATH [--max-images N]");
  }
  return result;
}

template <typename Row> std::int64_t maximum_gap(const std::vector<Row> &rows) {
  std::int64_t result = 0;
  for (std::size_t index = 1; index < rows.size(); ++index) {
    result = std::max(result, rows[index].timestamp_ns - rows[index - 1].timestamp_ns);
  }
  return result;
}

std::string json_escape(const std::string &value) {
  std::string result;
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      result.push_back('\\');
    }
    result.push_back(character);
  }
  return result;
}

int main(int argc, char **argv) {
  try {
    const Arguments args = parse_arguments(argc, argv);
    fs::create_directories(args.output);
    const auto imu = read_imu(args.mav0 / "imu0" / "data.csv");
    const auto camera = read_camera(args.mav0 / "cam0" / "data.csv", args.mav0 / "cam0" / "data");
    const std::string sequence = args.mav0.filename() == "mav0" ? args.mav0.parent_path().filename().string()
                                                                : args.mav0.filename().string();

    auto parser = std::make_shared<ov_core::YamlParser>(args.config.string());
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);
    ov_msckf::VioManagerOptions params;
    params.print_and_load(parser);
    params.num_opencv_threads = 0;
    params.use_multi_threading_pubs = false;
    params.use_multi_threading_subs = false;
    if (!parser->successful()) {
      throw std::runtime_error("OpenVINS configuration parsing failed");
    }
    if (params.use_stereo || params.state_options.num_cameras != 1) {
      throw std::runtime_error("runner requires use_stereo=false and max_cameras=1");
    }
    if (params.state_options.max_slam_features != 0) {
      throw std::runtime_error("clean MSCKF forbids persistent SLAM features (max_slam must be 0)");
    }

    auto manager = std::make_shared<ov_msckf::VioManager>(params);
    OpenVinsStateExporter state_exporter(args.output);
    OpenVinsFeatureExporter feature_exporter(args.output, sequence);

    const auto wall_start = std::chrono::steady_clock::now();
    const std::int64_t first_sensor_ns = std::min(imu.front().timestamp_ns, camera.front().timestamp_ns);
    std::int64_t initialized_at_ns = -1;
    std::int64_t initialization_decision_ns = -1;
    std::size_t initialization_attempt_frames = 0;
    Eigen::Vector4d initial_q_GtoI = Eigen::Vector4d::Zero();
    Eigen::Vector3d initial_bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d initial_ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d initial_gravity_in_body = Eigen::Vector3d::Zero();
    std::size_t imu_index = 0;
    std::size_t fed_imu = 0;
    std::size_t fed_camera = 0;
    std::size_t images_to_process = std::min(camera.size(), args.max_images);
    for (std::size_t camera_index = 0; camera_index < images_to_process; ++camera_index) {
      const auto &frame = camera[camera_index];
      while (imu_index < imu.size() && imu[imu_index].timestamp_ns <= frame.timestamp_ns) {
        ov_core::ImuData message;
        message.timestamp = static_cast<double>(imu[imu_index].timestamp_ns) * 1e-9;
        message.wm = imu[imu_index].gyro;
        message.am = imu[imu_index].accel;
        manager->feed_measurement_imu(message);
        ++imu_index;
        ++fed_imu;
      }
      if (imu_index >= imu.size()) {
        throw std::runtime_error("camera frame lacks a strictly later IMU sample");
      }
      // OpenVINS propagation buffers require one sample after the image time.
      ov_core::ImuData lookahead;
      lookahead.timestamp = static_cast<double>(imu[imu_index].timestamp_ns) * 1e-9;
      lookahead.wm = imu[imu_index].gyro;
      lookahead.am = imu[imu_index].accel;
      manager->feed_measurement_imu(lookahead);
      ++imu_index;
      ++fed_imu;

      cv::Mat image = cv::imread(frame.image.string(), cv::IMREAD_GRAYSCALE);
      if (image.empty()) {
        throw std::runtime_error("OpenCV failed to decode " + frame.image.string());
      }
      ov_core::CameraData message;
      message.timestamp = static_cast<double>(frame.timestamp_ns) * 1e-9;
      message.sensor_ids = {0};
      message.images = {image};
      message.masks = {cv::Mat::zeros(image.size(), CV_8UC1)};
      manager->feed_measurement_camera(message);
      ++fed_camera;
      if (!manager->initialized()) {
        ++initialization_attempt_frames;
      }
      if (manager->initialized() && initialized_at_ns < 0) {
        initialized_at_ns = static_cast<std::int64_t>(std::llround(manager->initialized_time() * 1e9));
        initialization_decision_ns = frame.timestamp_ns;
        const auto initialized_state = manager->get_state();
        initial_q_GtoI = initialized_state->_imu->quat();
        initial_bg = initialized_state->_imu->bias_g();
        initial_ba = initialized_state->_imu->bias_a();
        initial_gravity_in_body =
            initialized_state->_imu->Rot() * Eigen::Vector3d(0.0, 0.0, params.gravity_mag);
      }
      state_exporter.export_camera_update(frame.timestamp_ns, manager);
      feature_exporter.export_camera_update(frame.timestamp_ns, image.cols, image.rows, manager);
    }
    while (imu_index < imu.size()) {
      ov_core::ImuData message;
      message.timestamp = static_cast<double>(imu[imu_index].timestamp_ns) * 1e-9;
      message.wm = imu[imu_index].gyro;
      message.am = imu[imu_index].accel;
      manager->feed_measurement_imu(message);
      ++imu_index;
      ++fed_imu;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double runtime_s = std::chrono::duration<double>(wall_end - wall_start).count();
    const double processed_duration_s =
        (camera[images_to_process - 1].timestamp_ns - first_sensor_ns) * 1e-9;
    const auto state = manager->get_state();
    const std::string initialization_method =
        params.try_zupt ? "robust_static_beginning_only_zupt"
                        : (params.init_options.init_dyn_use ? "dynamic_fallback_enabled" : "robust_static_jerk");
    const std::string initialization_reason =
        params.try_zupt ? "same locked sensor-only rule waits for a visually and inertially stationary interval"
                        : "locked sensor-only OpenVINS initializer configuration";

    std::ofstream initialization(args.output / "initialization_report.json");
    initialization << std::setprecision(17)
                   << "{\n  \"truth_free\": true,\n  \"method\": \""
                   << initialization_method << "\",\n"
                   << "  \"success\": " << (manager->initialized() ? "true" : "false") << ",\n"
                   << "  \"initialization_timestamp_ns\": " << initialized_at_ns << ",\n"
                   << "  \"decision_camera_timestamp_ns\": " << initialization_decision_ns << ",\n"
                   << "  \"initialization_duration_sensor_s\": "
                   << (initialization_decision_ns >= 0 ? (initialization_decision_ns - first_sensor_ns) * 1e-9 : -1.0) << ",\n"
                   << "  \"preinitialization_camera_attempts\": " << initialization_attempt_frames << ",\n"
                   << "  \"gyro_bias_rad_s\": [" << initial_bg(0) << ", " << initial_bg(1) << ", " << initial_bg(2) << "],\n"
                   << "  \"accelerometer_bias_m_s2\": [" << initial_ba(0) << ", " << initial_ba(1) << ", " << initial_ba(2) << "],\n"
                   << "  \"q_GtoI_jpl_xyzw\": [" << initial_q_GtoI(0) << ", " << initial_q_GtoI(1) << ", "
                   << initial_q_GtoI(2) << ", " << initial_q_GtoI(3) << "],\n"
                   << "  \"gravity_in_initial_body_m_s2\": [" << initial_gravity_in_body(0) << ", "
                   << initial_gravity_in_body(1) << ", " << initial_gravity_in_body(2) << "],\n"
                   << "  \"reason\": \"" << json_escape(initialization_reason) << "\"\n}\n";

    std::ofstream manifest(args.output / "sequence_manifest.json");
    manifest << std::setprecision(17)
             << "{\n  \"sequence\": \"" << json_escape(sequence) << "\",\n"
             << "  \"mav0\": \"" << json_escape(fs::absolute(args.mav0).string()) << "\",\n"
             << "  \"config\": \"" << json_escape(fs::absolute(args.config).string()) << "\",\n"
             << "  \"total_imu_measurements\": " << imu.size() << ",\n"
             << "  \"total_images\": " << camera.size() << ",\n"
             << "  \"fed_imu_measurements\": " << fed_imu << ",\n"
             << "  \"fed_images\": " << fed_camera << ",\n"
             << "  \"first_timestamp_ns\": " << first_sensor_ns << ",\n"
             << "  \"last_timestamp_ns\": " << std::max(imu.back().timestamp_ns, camera.back().timestamp_ns) << ",\n"
             << "  \"dropped_measurements\": 0,\n  \"duplicate_timestamps\": 0,\n"
             << "  \"maximum_imu_gap_ns\": " << maximum_gap(imu) << ",\n"
             << "  \"maximum_camera_gap_ns\": " << maximum_gap(camera) << ",\n"
             << "  \"runtime_s\": " << runtime_s << ",\n"
             << "  \"processed_duration_s\": " << processed_duration_s << ",\n"
             << "  \"real_time_factor\": " << (runtime_s / std::max(processed_duration_s, 1e-12)) << "\n}\n";

    if (!manager->initialized()) {
      std::cerr << "truth-free initialization failed\n";
      return 2;
    }
    std::cout << "completed " << sequence << ": " << fed_imu << " IMU, " << fed_camera << " images, " << runtime_s << " s\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "clean MSCKF runner error: " << error.what() << '\n';
    return 1;
  }
}
