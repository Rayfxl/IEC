#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <chrono>
#include <memory>
#include <thread>
#include <algorithm>
#include <fstream>
#include <sys/resource.h>
using namespace MNN;
using namespace MNN::CV;
using namespace cv;

struct PerformanceMetrics {
    double preprocessing_time_ms{0.0};
    double inference_time_ms{0.0};
    double postprocessing_time_ms{0.0};
    size_t memory_usage_kb{0};
};

// 分类结果
struct ClassificationResult {
    size_t index;
    float score;
    bool operator<(const ClassificationResult& other) const {
        return score > other.score; // 降序排序
    }
};

// 获取TopK分类结果
std::vector<ClassificationResult> getTopK(const std::vector<float>& scores, int k) {
    std::vector<ClassificationResult> results;
    for (size_t i = 0; i < scores.size(); ++i) {
        results.push_back({i, scores[i]});
    }
    std::partial_sort(results.begin(), results.begin() + k, results.end());
    results.resize(k);
    return results;
}

// 获取当前进程内存使用情况
static size_t getCurrentMemoryUsage() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    size_t result = 0;
    char line[128];
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &result);
            break;
        }
    }
    fclose(file);
    return result;
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s model.mnn input.jpg\n", argv[0]);
        return -1;
    }

    const int num_runs = 10;
    std::vector<PerformanceMetrics> metrics_vec;
    std::vector<float> cls_scores;

    // 创建推理器
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]), Interpreter::destroy);
    if (!net) {
        printf("Failed to load model\n");
        return -1;
    }

    // 设置推理配置
    ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = std::thread::hardware_concurrency(); // 设置线程数
    BackendConfig backendConfig;
    backendConfig.memory = BackendConfig::Memory_High;
    backendConfig.power = BackendConfig::Power_High;
    backendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;

    auto session = net->createSession(config);
    auto input = net->getSessionInput(session, nullptr);
    auto output = net->getSessionOutput(session, nullptr);

    // 检查输入尺寸并修正
    std::vector<int> inputShape = input->shape();
    const int batch = inputShape[0];
    const int channel = inputShape[1];
    const int inputHeight = inputShape[2];
    const int inputWidth = inputShape[3];

    printf("Input shape: [%d, %d, %d, %d]\n", batch, channel, inputHeight, inputWidth);

    // 预处理配置
    ImageProcess::Config preprocess_config;
    preprocess_config.filterType = BILINEAR;
    preprocess_config.sourceFormat = BGR;
    preprocess_config.destFormat = RGB;
    preprocess_config.wrap = CLAMP_TO_EDGE;

    // 预处理归一化参数
    const float means[4] = {103.94f, 116.78f, 123.68f, 0.0f};
    const float norms[4] = {0.017f, 0.017f, 0.017f, 1.0f};
    memcpy(preprocess_config.mean, means, sizeof(means));
    memcpy(preprocess_config.normal, norms, sizeof(norms));

    // 创建预处理对象
    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(preprocess_config));

    // 记录基线内存
    size_t base_memory = getCurrentMemoryUsage();
    printf("Initial memory: %zu KB\n", base_memory);

    // 性能统计变量
    double min_preprocess = std::numeric_limits<double>::max();
    double max_preprocess = 0.0;
    double min_inference = std::numeric_limits<double>::max();
    double max_inference = 0.0;
    double min_postprocess = std::numeric_limits<double>::max();
    double max_postprocess = 0.0;
    double min_fps = std::numeric_limits<double>::max();
    double max_fps = 0.0;
    double total_fps = 0.0;  // 新增：用于计算平均 FPS
    size_t min_memory = std::numeric_limits<size_t>::max();
    size_t max_memory = 0;

    // 性能测试循环
    for (int i = 0; i < num_runs; i++) {
        PerformanceMetrics metrics;

        // 图像预处理
        auto start_preprocess = std::chrono::high_resolution_clock::now();
        cv::Mat img = imread(argv[2]);
        if (img.empty()) {
            printf("Failed to load image\n");
            return -1;
        }
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(inputWidth, inputHeight));

        Matrix trans;
        trans.setScale(1.0f, 1.0f); // 可以根据需要进行修改
        pretreat->setMatrix(trans);

        std::shared_ptr<Tensor> inputUser(new Tensor(input, Tensor::TENSORFLOW));
        pretreat->convert((uint8_t*)resized.data, resized.cols, resized.rows, 
                          resized.step[0], inputUser->host<uint8_t>(), 
                          inputWidth, inputHeight, 3, 0, inputUser->getType());
        input->copyFromHostTensor(inputUser.get());

        auto end_preprocess = std::chrono::high_resolution_clock::now();
        metrics.preprocessing_time_ms = std::chrono::duration<double, std::milli>(
            end_preprocess - start_preprocess).count();

        // 推理
        auto start_inference = std::chrono::high_resolution_clock::now();
        net->runSession(session);
        auto end_inference = std::chrono::high_resolution_clock::now();
        metrics.inference_time_ms = std::chrono::duration<double, std::milli>(
            end_inference - start_inference).count();

        // 后处理
        auto start_postprocess = std::chrono::high_resolution_clock::now();
        std::shared_ptr<Tensor> outputUser(new Tensor(output, Tensor::TENSORFLOW));
        output->copyToHostTensor(outputUser.get());

        cls_scores.resize(outputUser->elementSize());
        memcpy(cls_scores.data(), outputUser->host<float>(), 
               sizeof(float) * outputUser->elementSize());

        auto end_postprocess = std::chrono::high_resolution_clock::now();
        metrics.postprocessing_time_ms = std::chrono::duration<double, std::milli>(
            end_postprocess - start_postprocess).count();

        // 计算内存使用
        size_t current_memory = getCurrentMemoryUsage();
        metrics.memory_usage_kb = current_memory;

        min_memory = std::min(min_memory, current_memory);
        max_memory = std::max(max_memory, current_memory);

        // 计算FPS
        double current_fps = 1000.0 / (metrics.preprocessing_time_ms + 
                                       metrics.inference_time_ms + 
                                       metrics.postprocessing_time_ms);
        min_fps = std::min(min_fps, current_fps);
        max_fps = std::max(max_fps, current_fps);
        total_fps += current_fps;  // 累加 FPS

        // 更新min/max统计数据
        min_preprocess = std::min(min_preprocess, metrics.preprocessing_time_ms);
        max_preprocess = std::max(max_preprocess, metrics.preprocessing_time_ms);
        min_inference = std::min(min_inference, metrics.inference_time_ms);
        max_inference = std::max(max_inference, metrics.inference_time_ms);
        min_postprocess = std::min(min_postprocess, metrics.postprocessing_time_ms);
        max_postprocess = std::max(max_postprocess, metrics.postprocessing_time_ms);
        metrics_vec.push_back(metrics);
    }

    // 计算平均性能指标
    PerformanceMetrics avg_metrics;
    for (const auto& metrics : metrics_vec) {
        avg_metrics.preprocessing_time_ms += metrics.preprocessing_time_ms;
        avg_metrics.inference_time_ms += metrics.inference_time_ms;
        avg_metrics.postprocessing_time_ms += metrics.postprocessing_time_ms;
        avg_metrics.memory_usage_kb += metrics.memory_usage_kb;
    }
    avg_metrics.preprocessing_time_ms /= num_runs;
    avg_metrics.inference_time_ms /= num_runs;
    avg_metrics.postprocessing_time_ms /= num_runs;
    avg_metrics.memory_usage_kb /= num_runs;

    // 计算平均FPS
    double avg_fps = total_fps / num_runs;
    // 输出性能报告
    printf("\n=== Performance Report (%d runs) ===\n", num_runs);
    printf("Min/Max/Avg Preprocessing Time: %.2f/%.2f/%.2f ms\n", 
           min_preprocess, max_preprocess, avg_metrics.preprocessing_time_ms);
    printf("Min/Max/Avg Inference Time: %.2f/%.2f/%.2f ms\n",
           min_inference, max_inference, avg_metrics.inference_time_ms);
    printf("Min/Max/Avg Postprocessing Time: %.2f/%.2f/%.2f ms\n",
           min_postprocess, max_postprocess, avg_metrics.postprocessing_time_ms);
    printf("Memory Usage: Base=%zu KB, Peak=%zu KB, Growth=%zu KB\n",
       base_memory, max_memory, max_memory - base_memory);
    printf("Min/Max/Avg FPS: %.2f/%.2f/%.2f\n", min_fps, max_fps, avg_fps);

    // 输出TopK分类结果
    int topK = 3;
    auto topKResults = getTopK(cls_scores, topK);
    printf("\nTop %d Results:\n", topK);
    for (size_t i = 0; i < topKResults.size(); ++i) {
        printf("Class: %zu, Score: %.4f\n", topKResults[i].index, topKResults[i].score);
    }
    return 0;
}
