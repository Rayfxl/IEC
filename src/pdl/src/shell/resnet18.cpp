#include <paddle_api.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sstream>         
#include <vector>               

int WARMUP_COUNT = 1;
int REPEAT_COUNT = 5;
const int CPU_THREAD_NUM = 1;
// 添加全局常量定义
const int TOPK = 3;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_NO_BIND;


inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

template <typename T>
void get_value_from_sstream(std::stringstream *ss, T *value) {
  (*ss) >> (*value);
}

template <>
void get_value_from_sstream<std::string>(std::stringstream *ss,
                                         std::string *value) {
  *value = ss->str();
}

template <typename T>
std::vector<T> split_string(const std::string &str, char sep) {
  std::stringstream ss;
  std::vector<T> values;
  T value;
  values.clear();
  for (auto c : str) {
    if (c != sep) {
      ss << c;
    } else {
      get_value_from_sstream<T>(&ss, &value);
      values.push_back(std::move(value));
      ss.str({});
      ss.clear();
    }
  }
  if (!ss.str().empty()) {
    get_value_from_sstream<T>(&ss, &value);
    values.push_back(std::move(value));
    ss.str({});
    ss.clear();
  }
  return values;
}

bool read_file(const std::string &filename,
               std::vector<char> *contents,
               bool binary = true) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}


int64_t shape_production(std::vector<int64_t> shape) {
  int64_t s = 1;
  for (int64_t dim : shape) {
    s *= dim;
  }
  return s;
}

void nhwc32nc3hw(const float *src,
                 float *dst,
                 const float *mean,
                 const float *std,
                 int width,
                 int height) {
  int size = height * width;
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;

  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) / std[0];
    *(dst_c1++) = (*(src++) - mean[1]) / std[1];
    *(dst_c2++) = (*(src++) - mean[2]) / std[2];
  }
}

typedef struct {
  int width;
  int height;
  std::vector<float> mean;
  std::vector<float> std;
} CONFIG;


CONFIG load_config(const std::string &path) {
  CONFIG config;
  std::vector<char> buffer;
  if (!read_file(path, &buffer, false)) {
    printf("Failed to load the config file %s\n", path.c_str());
    exit(-1);
  }
  std::string dir = ".";
  auto pos = path.find_last_of("/");
  if (pos != std::string::npos) {
    dir = path.substr(0, pos);
  }
  printf("dir: %s\n", dir.c_str());
  std::string content(buffer.begin(), buffer.end());
  auto lines = split_string<std::string>(content, '\n');
  std::map<std::string, std::string> values;
  for (auto &line : lines) {
    auto value = split_string<std::string>(line, ':');
    if (value.size() != 2) {
      printf("Format error at '%s', it should be '<key>:<value>'.\n",
             line.c_str());
      exit(-1);
    }
    values[value[0]] = value[1];
  }
  // width
  if (!values.count("width")) {
    printf("Missing the key 'width'!\n");
    exit(-1);
  }
  config.width = atoi(values["width"].c_str());
  if (config.width <= 0) {
    printf("The key 'width' should > 0, but receive %d!\n", config.width);
    exit(-1);
  }
  printf("width: %d\n", config.width);
  // height
  if (!values.count("height")) {
    printf("Missing the key 'height' !\n");
    exit(-1);
  }
  config.height = atoi(values["height"].c_str());
  if (config.height <= 0) {
    printf("The key 'height' should > 0, but receive %d!\n", config.height);
    exit(-1);
  }
  printf("height: %d\n", config.height);
  // mean
  if (!values.count("mean")) {
    printf("Missing the key 'mean'!\n");
    exit(-1);
  }
  config.mean = split_string<float>(values["mean"], ',');
  if (config.mean.size() != 3) {
    printf("The key 'mean' should contain 3 values, but receive %u!\n",
           config.mean.size());
    exit(-1);
  }
  printf("mean: %f,%f,%f\n", config.mean[0], config.mean[1], config.mean[2]);
  // std
  if (!values.count("std")) {
    printf("Missing the key 'std' !\n");
    exit(-1);
  }
  config.std = split_string<float>(values["std"], ',');
  if (config.std.size() != 3) {
    printf("The key 'std' should contain 3 values, but receive %u!\n",
           config.std.size());
    exit(-1);
  }
  printf("std: %f,%f,%f\n", config.std[0], config.std[1], config.std[2]);
  return config;
}

std::vector<std::string> load_dataset(const std::string &path) {
  std::vector<char> buffer;
  if (!read_file(path, &buffer, false)) {
    printf("Failed to load the dataset list file %s\n", path.c_str());
    exit(-1);
  }
  std::string content(buffer.begin(), buffer.end());
  auto lines = split_string<std::string>(content, '\n');
  if (lines.empty()) {
    printf("The dataset list file %s should not be empty!\n", path.c_str());
    exit(-1);
  }
  return lines;
}

size_t getCurrentMemoryUsage() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    
    char line[128];
    size_t vm_rss = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &vm_rss);
            break;
        }
    }
    fclose(file);
    return vm_rss;
}

void process(std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor,
             const std::string &config_path,
             const std::string &dataset_dir) {
  // 声明在函数开始处,并在整个函数中保持有效
  std::vector<std::pair<float, int>> final_sorted_data;

  float cur_costs[3] = {0.0f, 0.0f, 0.0f};  // 预处理、推理、后处理
  float total_costs[3] = {0.0f, 0.0f, 0.0f};
  float max_costs[3] = {0.0f, 0.0f, 0.0f};
  float min_costs[3] = {std::numeric_limits<float>::max(), 
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max()};
  // Parse the config file to extract the model info
  auto config = load_config(config_path);
  // Load dataset list
  auto dataset = load_dataset(dataset_dir + "/list.txt");
  // Prepare for inference
  std::unique_ptr<paddle::lite_api::Tensor> image_tensor =
      predictor->GetInput(0);
  image_tensor->Resize({1, 3, config.height, config.width});
  auto image_data = image_tensor->mutable_data<float>();
  // 只使用第一张图片
  auto sample_name = dataset[0];
  auto input_path = dataset_dir + "/inputs/" + sample_name;
  predictor->Run();  // Warmup
  // Traverse the list of the dataset and run inference on each sample
  
  size_t base_memory = getCurrentMemoryUsage();
  size_t peak_memory = base_memory;
  
  const int num_runs = 10; // 设置运行次数
  int iter_count = 0;

  // auto sample_count = dataset.size();
  for (size_t i = 0; i < num_runs; i++) {
    // auto sample_name = dataset[i];
    // printf("[%u/%u] Processing %s\n", i + 1, sample_count, sample_name.c_str());
    // auto input_path = dataset_dir + "/inputs/" + sample_name;
    auto output_path = dataset_dir + "/outputs/" + sample_name;
    // // Check if input and output is accessable
    // if (access(input_path.c_str(), R_OK) != 0) {
    //   printf("%s not found or readable!\n", input_path.c_str());
    //   exit(-1);
    // }
    // Preprocess
    double start = get_current_us();
    // image tensor
    cv::Mat origin_image = cv::imread(input_path);
    cv::Mat resized_image;
    cv::resize(origin_image,
               resized_image,
               cv::Size(config.width, config.height),
               0,
               0);
    if (resized_image.channels() == 3) {
      cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    } else if (resized_image.channels() == 4) {
      cv::cvtColor(resized_image, resized_image, cv::COLOR_BGRA2RGB);
    } else {
      printf("The channel size should be 4 or 3, but receive %d!\n",
             resized_image.channels());
      exit(-1);
    }
    resized_image.convertTo(resized_image, CV_32FC3, 1 / 255.f);
    nhwc32nc3hw(reinterpret_cast<const float *>(resized_image.data),
                image_data,
                config.mean.data(),
                config.std.data(),
                config.width,
                config.height);
    double end = get_current_us();
    cur_costs[0] = (end - start) / 1000.0f;
    // Inference
    start = get_current_us();
    predictor->Run();
    end = get_current_us();
    cur_costs[1] = (end - start) / 1000.0f;
    // Postprocess
    start = get_current_us();
    auto output_tensor = predictor->GetOutput(0);
    auto output_data = output_tensor->data<float>();
    auto output_size = shape_production(output_tensor->shape());
    std::vector<std::pair<float, int>> sorted_data;
    for (int64_t j = 0; j < output_size; j++) {
      sorted_data.push_back(std::make_pair(output_data[j], j));
    }
    // const int TOPK = 3;
     // 在最后一次迭代时保存sorted_data
     // 修改保存最后结果的逻辑
    if(i == num_runs - 1) {
        final_sorted_data = sorted_data;  // 保存到函数级变量
        std::partial_sort(final_sorted_data.begin(),
                         final_sorted_data.begin() + TOPK,
                         final_sorted_data.end(),
                         [&](std::pair<float, int> a, std::pair<float, int> b) {
                             return (a.first > b.first);
                         });
    }
    // std::partial_sort(sorted_data.begin(),
    //                   sorted_data.begin() + TOPK,
    //                   sorted_data.end(),
    //                   [&](std::pair<float, int> a, std::pair<float, int> b) {
    //                     return (a.first > b.first);
    //                   });
    for (int j = 0; j < TOPK; j++) {
      auto class_id = sorted_data[j].second;
      auto score = sorted_data[j].first;
        printf("Class: %d, Score: %.6f\n", class_id, score);
        
        // 在图像上绘制文本也相应修改
        cv::putText(origin_image,
                    "Class:" + std::to_string(class_id) + " Score:" + 
                        std::to_string(score),
                    cv::Point2d(5, j * 18 + 20),
                    cv::FONT_HERSHEY_PLAIN,
                    1,
                    cv::Scalar(51, 255, 255));
    }
    cv::imwrite(output_path, origin_image);
    end = get_current_us();
    cur_costs[2] = (end - start) / 1000.0f;
    // Update statistics
    for(int j = 0; j < 3; j++) {
        total_costs[j] += cur_costs[j];
        max_costs[j] = std::max(max_costs[j], cur_costs[j]); 
        min_costs[j] = std::min(min_costs[j], cur_costs[j]);
    }
    printf(
        "[%d] Preprocess time: %f ms Prediction time: %f ms Postprocess time: "
        "%f ms\n",
        iter_count,
        cur_costs[0],
        cur_costs[1],
        cur_costs[2]);
    
    // Memory tracking
    size_t current_memory = getCurrentMemoryUsage();
    peak_memory = std::max(peak_memory, current_memory);
    iter_count++;
    // // 每次迭代结束时更新内存使用峰值
    // size_t current_memory = getCurrentMemoryUsage();
    // peak_memory = std::max(peak_memory, current_memory);
  }
  // 修改性能输出部分
  printf("\n=== Performance Report (%d runs) ===\n", num_runs);
  printf("Min/Max/Avg Preprocessing Time: %.2f/%.2f/%.2f ms\n",
          min_costs[0], max_costs[0], total_costs[0] / iter_count);
  printf("Min/Max/Avg Inference Time: %.2f/%.2f/%.2f ms\n",
          min_costs[1], max_costs[1], total_costs[1] / iter_count);
  printf("Min/Max/Avg Postprocessing Time: %.2f/%.2f/%.2f ms\n",
          min_costs[2], max_costs[2], total_costs[2] / iter_count);
  
  size_t memory_growth = peak_memory - base_memory;
  printf("Memory Usage: Base=%zu KB, Peak=%zu KB, Growth=%zu KB\n",
          base_memory, peak_memory, memory_growth);
          
  double total_time = (total_costs[0] + total_costs[1] + total_costs[2]) / iter_count;
  printf("Min/Max/Avg FPS: %.2f/%.2f/%.2f\n",
          1000.0 / (max_costs[0] + max_costs[1] + max_costs[2]),
          1000.0 / (min_costs[0] + min_costs[1] + min_costs[2]),
          1000.0 / total_time);

  // 使用final_sorted_data输出结果
  printf("\nTop %d Results:\n", TOPK);
  for (int j = 0; j < TOPK; j++) {
      auto class_id = final_sorted_data[j].second;
      auto score = final_sorted_data[j].first;
      printf("Class: %d, Score: %.6f\n", class_id, score);
  }
}

int main(int argc, char **argv) {

  // 解析参数
  std::string model_dir = argv[1];
  std::string config_path = argv[2];
  std::string dataset_dir = argv[3];
  std::vector<std::string> nnadapter_device_names = 
      split_string<std::string>(argv[4], ',');

  // 配置预测器
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(model_dir + ".nb");
  config.set_threads(CPU_THREAD_NUM);
  config.set_power_mode(CPU_POWER_MODE);

  // 创建预测器
  try {
    auto predictor = 
        paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
            config);
    process(predictor, config_path, dataset_dir);
  } catch (std::exception e) {
    printf("An internal error occurred in PaddleLite.\n");
    return -1;
  }

  return 0;
}
