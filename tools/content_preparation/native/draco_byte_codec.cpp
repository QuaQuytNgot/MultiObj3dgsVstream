// Project lossless byte-attribute bridge over public Draco. This is not the
// unpublished enhanced-Draco implementation described by LTS.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "draco/compression/decode.h"
#include "draco/compression/expert_encode.h"
#include "draco/core/draco_version.h"
#include "draco/core/encoder_buffer.h"
#include "draco/point_cloud/point_cloud_builder.h"

namespace {
constexpr size_t kComponents = 64;
constexpr size_t kMaxBytes = size_t(2) * 1024 * 1024 * 1024;

std::vector<uint8_t> Read(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open input: " + path);
  const auto length = file.tellg();
  if (length < 0 || static_cast<uint64_t>(length) > kMaxBytes)
    throw std::runtime_error("Input exceeds the bridge's 2 GiB stream limit");
  std::vector<uint8_t> data(static_cast<size_t>(length));
  file.seekg(0);
  if (!data.empty() && !file.read(reinterpret_cast<char *>(data.data()), data.size()))
    throw std::runtime_error("Failed reading complete input");
  return data;
}

void Write(const std::string &path, const void *data, size_t length) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file || !file.write(static_cast<const char *>(data), length))
    throw std::runtime_error("Cannot write output: " + path);
}

uint64_t Number(const std::string &value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Expected unsigned decimal integer");
  return std::stoull(value);
}
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "{\"bridge\":\"draco_byte_codec\",\"bridge_version\":\"1.0.0\","
                << "\"draco_version\":\"" << draco::kDracoVersion << "\","
                << "\"draco_commit\":\"" << CP_DRACO_COMMIT << "\","
                << "\"compiler\":\"" << CP_COMPILER << "\","
                << "\"build_type\":\"" << CP_BUILD_TYPE << "\","
                << "\"method\":\"sequential\",\"prediction\":\"none\","
                << "\"attribute_type\":\"DT_UINT8\",\"maximum_components\":64,"
                << "\"deduplication\":false,\"quantization\":false}\n";
      return 0;
    }
    if (argc < 2) throw std::runtime_error("Usage: encode|decode --input PATH --output PATH --rows N --row-bytes B [--encoder-speed 3 --decoder-speed 3]");
    const std::string operation(argv[1]);
    if (operation != "encode" && operation != "decode") throw std::runtime_error("Unknown operation");
    std::map<std::string, std::string> options;
    const std::vector<std::string> allowed = {"--input", "--output", "--rows", "--row-bytes", "--encoder-speed", "--decoder-speed"};
    for (int i = 2; i < argc; i += 2) {
      const std::string key(argv[i]);
      if (i + 1 >= argc || std::find(allowed.begin(), allowed.end(), key) == allowed.end() || !options.emplace(key, argv[i + 1]).second)
        throw std::runtime_error("Unknown, duplicate or incomplete argument: " + key);
    }
    for (const auto &key : {"--input", "--output", "--rows", "--row-bytes"})
      if (!options.count(key)) throw std::runtime_error(std::string("Missing ") + key);
    const uint64_t rows = Number(options.at("--rows"));
    const uint64_t row_bytes = Number(options.at("--row-bytes"));
    if (!rows || !row_bytes || row_bytes > 4096 || rows > uint64_t(std::numeric_limits<int32_t>::max()) / kComponents || rows > kMaxBytes / row_bytes)
      throw std::runtime_error("Invalid stream dimensions (nonempty, <=4096 bytes/row, <=2 GiB)");
    const size_t expected_bytes = rows * row_bytes;
    const auto data = Read(options.at("--input"));
    if (operation == "encode") {
      if (data.size() != expected_bytes) throw std::runtime_error("Raw byte count does not match dimensions");
      const uint64_t encoder_speed = options.count("--encoder-speed") ? Number(options.at("--encoder-speed")) : 3;
      const uint64_t decoder_speed = options.count("--decoder-speed") ? Number(options.at("--decoder-speed")) : 3;
      if (encoder_speed > 10 || decoder_speed > 10) throw std::runtime_error("Speed must be in 0..10");
      draco::PointCloudBuilder builder;
      builder.Start(static_cast<uint32_t>(rows));
      std::vector<int> attribute_ids;
      for (size_t offset = 0; offset < row_bytes; offset += kComponents) {
        const int components = static_cast<int>(std::min(kComponents, row_bytes - offset));
        const int attribute = builder.AddAttribute(draco::GeometryAttribute::GENERIC, components, draco::DT_UINT8);
        if (attribute < 0) throw std::runtime_error("Cannot allocate Draco byte attribute");
        builder.SetAttributeUniqueId(attribute, static_cast<uint32_t>(offset / kComponents));
        builder.SetAttributeValuesForAllPoints(attribute, data.data() + offset, static_cast<int>(row_bytes));
        attribute_ids.push_back(attribute);
      }
      auto cloud = builder.Finalize(false);
      if (!cloud || cloud->num_points() != rows) throw std::runtime_error("Unexpected deduplication");
      draco::ExpertEncoder encoder(*cloud);
      encoder.SetEncodingMethod(draco::POINT_CLOUD_SEQUENTIAL_ENCODING);
      encoder.SetSpeedOptions(encoder_speed, decoder_speed);
      encoder.SetUseBuiltInAttributeCompression(true);
      for (const int attribute : attribute_ids) {
        const auto prediction = encoder.SetAttributePredictionScheme(attribute, draco::PREDICTION_NONE);
        if (!prediction.ok()) throw std::runtime_error(prediction.error_msg_string());
      }
      draco::EncoderBuffer buffer;
      const auto status = encoder.EncodeToBuffer(&buffer);
      if (!status.ok()) throw std::runtime_error(status.error_msg_string());
      Write(options.at("--output"), buffer.data(), buffer.size());
    } else {
      draco::DecoderBuffer buffer;
      buffer.Init(reinterpret_cast<const char *>(data.data()), data.size());
      draco::Decoder decoder;
      auto result = decoder.DecodePointCloudFromBuffer(&buffer);
      if (!result.ok()) throw std::runtime_error(result.status().error_msg_string());
      auto cloud = std::move(result).value();
      if (cloud->num_points() != rows || cloud->num_attributes() != int((row_bytes + kComponents - 1) / kComponents) || buffer.remaining_size())
        throw std::runtime_error("Decoded count/schema or trailing-byte mismatch");
      std::vector<uint8_t> raw(expected_bytes);
      for (size_t offset = 0; offset < row_bytes; offset += kComponents) {
        const int components = std::min(kComponents, row_bytes - offset);
        const auto *attribute = cloud->GetAttributeByUniqueId(offset / kComponents);
        if (!attribute || attribute->attribute_type() != draco::GeometryAttribute::GENERIC || attribute->data_type() != draco::DT_UINT8 || attribute->normalized() || attribute->num_components() != components)
          throw std::runtime_error("Wrong byte-attribute schema");
        for (uint32_t point = 0; point < rows; ++point)
          attribute->GetValue(attribute->mapped_index(draco::PointIndex(point)), raw.data() + point * row_bytes + offset);
      }
      Write(options.at("--output"), raw.data(), raw.size());
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draco_byte_codec: " << error.what() << '\n';
    return 1;
  }
}
