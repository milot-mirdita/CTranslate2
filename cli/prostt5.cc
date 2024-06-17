#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdint>

#include <cxxopts.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ctranslate2/types.h>
#include <ctranslate2/random.h>
#include <ctranslate2/devices.h>
#include <ctranslate2/encoder.h>
// #include <ctranslate2/utils.h>
// #include <ctranslate2/profiler.h>

struct CnnWeights {
  std::vector<ctranslate2::dim_t> shape;
  std::vector<float> values;
};

bool read_weights(const char* filename, std::vector<CnnWeights>& weights) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        return false;
    }
    size_t filesize = sb.st_size;

    void* map = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return false;
    }
    close(fd);

    char* data = static_cast<char*>(map);
    size_t offset = 0;
    while (offset < filesize) {
        int shape_arr[4];
        std::memcpy(shape_arr, data + offset, 4 * sizeof(int));
        offset += 4 * sizeof(int);

        int shape_size = 0;
        for (int i = 0; i < 4; ++i) {
            if (shape_arr[i] == -1) break;
            shape_size++;
        }
        std::vector<ctranslate2::dim_t> shape(shape_arr, shape_arr + shape_size);
        int num_elements = 1;
        for (int dim : shape) {
            num_elements *= dim;
        }

        std::vector<float> values(num_elements);
        std::memcpy(values.data(), data + offset, num_elements * sizeof(float));
        offset += num_elements * sizeof(float);

        weights.emplace_back(CnnWeights{std::move(shape), std::move(values)});
    }

    if (munmap(map, filesize) == -1) {
        return false;
    }
    return true;
}

std::vector<std::vector<uint8_t>> compute_argmax_per_batch(const ctranslate2::StorageView& tensor) {
  if (tensor.rank() != 3) {
    throw std::runtime_error("Input tensor must have 3 dimensions (batch_size, channels, seq_length).");
  }

  const ctranslate2::dim_t batch_size = tensor.dim(0);
  const ctranslate2::dim_t channels = tensor.dim(1);
  const ctranslate2::dim_t seq_length = tensor.dim(2);

  std::vector<float> flattened_tensor = tensor.to_vector<float>();
  std::vector<std::vector<uint8_t>> argmax_indices(batch_size, std::vector<uint8_t>(seq_length));
  for (ctranslate2::dim_t b = 0; b < batch_size; ++b) {
    for (ctranslate2::dim_t s = 0; s < seq_length; ++s) {
      float max_value = -std::numeric_limits<float>::infinity();
      uint8_t max_index = 0;
      for (ctranslate2::dim_t c = 0; c < channels; ++c) {
        float value = flattened_tensor[b * channels * seq_length + c * seq_length + s];
        if (value > max_value) {
          max_value = value;
          max_index = c;
        }
      }
      argmax_indices[b][s] = max_index;
    }
  }
  return argmax_indices;
}

char number_to_char(uint32_t n) {
    switch (n) {
        case 0:  return 'A';
        case 1:  return 'C';
        case 2:  return 'D';
        case 3:  return 'E';
        case 4:  return 'F';
        case 5:  return 'G';
        case 6:  return 'H';
        case 7:  return 'I';
        case 8:  return 'K';
        case 9:  return 'L';
        case 10: return 'M';
        case 11: return 'N';
        case 12: return 'P';
        case 13: return 'Q';
        case 14: return 'R';
        case 15: return 'S';
        case 16: return 'T';
        case 17: return 'V';
        case 18: return 'W';
        case 19: return 'Y';
        default: return 'X'; // Default case for numbers not in the list
    }
}

int main(int argc, char* argv[]) {
  cxxopts::Options cmd_options("ct2-translator", "CTranslate2 translator client");
  cmd_options.custom_help("--model <directory> [OPTIONS]");

  cmd_options.add_options("General")
    ("h,help", "Display available options.")
    ("seed", "Seed value of the random generators.",
     cxxopts::value<unsigned int>()->default_value("0"))
    ;

  cmd_options.add_options("Device")
    ("inter_threads", "Maximum number of CPU translations to run in parallel.",
     cxxopts::value<size_t>()->default_value("1"))
    ("intra_threads", "Number of computation threads (set to 0 to use the default value).",
     cxxopts::value<size_t>()->default_value("0"))
    ("device", "Device to use (can be cpu, cuda, auto).",
     cxxopts::value<std::string>()->default_value("cpu"))
    ("device_index", "Comma-separated list of device IDs to use.",
     cxxopts::value<std::vector<int>>()->default_value("0"))
    ("cpu_core_offset", "Pin worker threads to CPU cores starting from this offset.",
     cxxopts::value<int>()->default_value("-1"))
    ;

  cmd_options.add_options("Model")
    ("model", "Path to the CTranslate2 model directory.", cxxopts::value<std::string>())
    ("compute_type", "The type used for computation: default, auto, float32, float16, bfloat16, int16, int8, int8_float32, int8_float16, or int8_bfloat16",
     cxxopts::value<std::string>()->default_value("default"))
    ("cuda_compute_type", "Computation type on CUDA devices (overrides compute_type)",
     cxxopts::value<std::string>())
    ("cpu_compute_type", "Computation type on CPU devices (overrides compute_type)",
     cxxopts::value<std::string>())
    ;

  cmd_options.add_options("Data")
    ("src", "Path to the source file (read from the standard input if not set).",
     cxxopts::value<std::string>())
    ("out", "Path to the output file (write to the standard output if not set).",
     cxxopts::value<std::string>())
    ("batch_size", "Size of the batch to forward into the model at once.",
     cxxopts::value<size_t>()->default_value("32"))
    ("read_batch_size", "Size of the batch to read at once (defaults to batch_size).",
     cxxopts::value<size_t>()->default_value("0"))
    ("max_queued_batches", "Maximum number of batches to load in advance (set -1 for unlimited, 0 for an automatic value).",
     cxxopts::value<long>()->default_value("0"))
    ("batch_type", "Batch type (can be examples, tokens).",
     cxxopts::value<std::string>()->default_value("examples"))
    ("max_input_length", "Truncate inputs after this many tokens (set 0 to disable).",
     cxxopts::value<size_t>()->default_value("1024"))
    ;

  auto args = cmd_options.parse(argc, argv);
  if (args.count("help")) {
    std::cerr << cmd_options.help() << std::endl;
    return 0;
  }
  if (!args.count("model")) {
    throw std::invalid_argument("Option --model is required to run translation");
  }
  if (args.count("seed") != 0) {
    ctranslate2::set_random_seed(args["seed"].as<unsigned int>());
  }
  size_t inter_threads = args["inter_threads"].as<size_t>();
  size_t intra_threads = args["intra_threads"].as<size_t>();

  const auto device = ctranslate2::str_to_device(args["device"].as<std::string>());
  auto compute_type = ctranslate2::str_to_compute_type(args["compute_type"].as<std::string>());
  switch (device) {
  case ctranslate2::Device::CPU:
    if (args.count("cpu_compute_type")) {
      compute_type = ctranslate2::str_to_compute_type(args["cpu_compute_type"].as<std::string>());
    }
    break;
  case ctranslate2::Device::CUDA:
    if (args.count("cuda_compute_type")) {
      compute_type = ctranslate2::str_to_compute_type(args["cuda_compute_type"].as<std::string>());
    }
    break;
  };

  std::string model_dir = args["model"].as<std::string>();
  std::vector<CnnWeights> weights;
  std::string cnn_path = model_dir + "/weights.bin";
  if (read_weights(cnn_path.c_str(), weights) == false) {
      return EXIT_FAILURE;
  }
  auto cnnDtype = ctranslate2::DataType::FLOAT32;
  ctranslate2::StorageView conv0(weights[0].shape, cnnDtype, device);
  conv0.view(weights[0].values.data(), weights[0].shape);
  ctranslate2::StorageView bias0(weights[1].shape, cnnDtype, device);
  bias0.view(weights[1].values.data(), weights[1].shape);

  ctranslate2::StorageView conv1(weights[2].shape, cnnDtype, device);
  conv1.view(weights[2].values.data(), weights[2].shape);
  ctranslate2::StorageView bias1(weights[3].shape, cnnDtype, device);
  bias1.view(weights[3].values.data(), weights[3].shape);

  ctranslate2::ReplicaPoolConfig pool_config;
  pool_config.num_threads_per_replica = intra_threads;
  pool_config.max_queued_batches = args["max_queued_batches"].as<long>();
  pool_config.cpu_core_offset = args["cpu_core_offset"].as<int>();

  ctranslate2::models::ModelLoader model_loader(model_dir);
  model_loader.device = device;
  model_loader.device_indices = args["device_index"].as<std::vector<int>>();
  model_loader.compute_type = compute_type;
  model_loader.num_replicas_per_device = inter_threads;

  ctranslate2::Encoder enc(model_loader, pool_config);
  const auto* model = enc.get_first_replica().model().get();
  const ctranslate2::models::LanguageModel* lm = reinterpret_cast<const ctranslate2::models::LanguageModel*>(model);
  const auto& vocab = lm->get_vocabulary();
  // std::vector<size_t> seq = {
  //   149, 3, 17, 14, 11, 8, 9, 4, 22, 19, 14, 7, 4, 9, 20, 3, 14, 6, 10, 11, 7, 17, 9, 3, 8, 16, 10, 5, 12, 10, 4, 18, 14, 20, 19, 15, 1
  // };
  // std::string input_seq = "ANKTRELCMKSLEHAKVDTSNEARQDGIDLYKHMF";
  std::string input_seq = args["src"].as<std::string>();
  std::vector<size_t> seq;
  seq.emplace_back(149);
  for (const char& c : input_seq) {
    std::string curr(1, c);
    size_t id = vocab.to_id(curr, true);
    seq.emplace_back(id);
  }
  std::cout << std::endl;
  seq.emplace_back(1);
  std::vector<std::vector<size_t>> input;
  // for(int i = 0; i < 10; i++){
    input.push_back(seq);
  // }
  auto start = std::chrono::high_resolution_clock::now();
  auto bla = enc.forward_batch_async(input);
  auto ys = bla.get().last_hidden_state;
  // std::cout << "hidden:" << ys << std::endl;

  const ctranslate2::DataType dtype = ys.dtype();
  const ctranslate2::dim_t batch_size = ys.dim(0);
  const ctranslate2::dim_t seq_len = ys.dim(1);
  const ctranslate2::dim_t hidden_dim = ys.dim(2);

  // Use ops::Slide to slice the tensor along the third dimension (axis = 2) and remove the first element
  ctranslate2::ops::Slide slide_op(1, 1, seq_len - 2, /* no_copy= */ false);
  ctranslate2::StorageView sliced_ys(device, dtype);
  slide_op(ys, sliced_ys);
  // std::cout << "slide:" << sliced_ys << std::endl;

  ctranslate2::StorageView pad_right({1, 1, hidden_dim}, dtype, device);
  pad_right.zero();

  ctranslate2::ops::Concat cat_op(1);
  ctranslate2::StorageView padded(device, dtype);
  cat_op({&sliced_ys, &pad_right}, padded);
  // std::cout << "padded:" << padded << std::endl;

  ctranslate2::ops::Transpose transpose_op(std::vector<ctranslate2::dim_t>{{0, 2, 1}});
  ctranslate2::StorageView transposed(device, dtype);
  transpose_op(padded, transposed);
  transposed.expand_dims(3);
  // std::cout << "transposed:" << transposed << std::endl;

  ctranslate2::StorageView conv0_out(device, cnnDtype);
  ctranslate2::ops::Conv2D conv_op(1, 1, 3, 0, 1);
  conv_op(transposed, conv0, bias0, conv0_out, NULL);
  // std::cout << "conv0: " << conv0 << std::endl;
  // std::cout << "conv0_out: " << conv0_out << std::endl;

  ctranslate2::StorageView relu0_out(device, cnnDtype);
  ctranslate2::ops::ReLU relu_op;
  relu_op(conv0_out, relu0_out);
  // std::cout << "relu0_out: " << relu0_out << std::endl;

  conv_op(relu0_out, conv1, bias1, conv0_out, NULL);
  conv0_out.squeeze(-1);
  // std::cout << "conv1_out: " << conv0_out << std::endl;

  std::vector<std::vector<uint8_t>> argmax_results = compute_argmax_per_batch(conv0_out);
  for (const auto& batch : argmax_results) {
    std::string pred;
    for (const auto& index : batch) {
      pred.append(1, number_to_char(index));
    }
    std::cout << pred << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  return 0;
}
