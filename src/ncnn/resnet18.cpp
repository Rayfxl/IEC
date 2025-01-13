#include "net.h"
#include <algorithm>
#include <fstream>
#include <unistd.h>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <thread>

// static ncnn::Net resnet18;  // 改为全局静态变量
// static bool model_loaded = false;
class ModelManager {
public:
    static ModelManager& getInstance() {
        static ModelManager instance;
        return instance;
    }
    
    ncnn::Net& getModel() { return resnet18; }
    bool isLoaded() const { return model_loaded; }
    void setLoaded(bool loaded) { model_loaded = loaded; }
    
private:
    ModelManager() {
        // // 初始化GPU实例
        // ncnn::create_gpu_instance();
    }
    ~ModelManager() {
        try {
            // // 确保先关闭GPU加速
            // resnet18.opt.use_vulkan_compute = false;
            
            // 清除extractor
            resnet18.clear();
            
            // // 等待GPU资源释放
            // ncnn::destroy_gpu_instance();
            
        } catch (const std::exception& e) {
            fprintf(stderr, "Error during cleanup: %s\n", e.what());
        }
    }
    
    ncnn::Net resnet18;
    bool model_loaded{false};
};
struct PerformanceMetrics {
    double inference_time_ms;
    double preprocessing_time_ms;
    double postprocessing_time_ms;
    size_t memory_usage_kb;
};
static ncnn::PoolAllocator poolallocator;
static ncnn::UnlockedPoolAllocator blob_pool_allocator;
// 添加内存统计函数
static size_t getCurrentMemoryUsage() {
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

static bool initModel() {
    static std::once_flag init_flag;
    static bool init_success = false;
    std::call_once(init_flag, []() {
        auto& manager = ModelManager::getInstance();
        if (!manager.isLoaded()) {
            auto& resnet18 = manager.getModel();
            // // 检查GPU可用性
            // int gpu_count = ncnn::get_gpu_count();
            // printf("GPU Count: %d\n", gpu_count);
            // 设置内存池
            resnet18.opt.blob_allocator = &blob_pool_allocator;
            resnet18.opt.workspace_allocator = &poolallocator;

            // resnet18.opt.use_vulkan_compute = (gpu_count > 0);
            resnet18.opt.use_fp16_packed = true;
            resnet18.opt.use_fp16_storage = true;
            resnet18.opt.use_packing_layout = true;
            // resnet18.opt.use_bf16_storage = true;
            resnet18.opt.use_fp16_arithmetic = true;
            resnet18.opt.lightmode = true;
            resnet18.opt.use_winograd_convolution = true;
            resnet18.opt.use_sgemm_convolution = true;
            resnet18.opt.use_int8_inference = true;
            resnet18.opt.num_threads = 4; // 自动设置线程数;
            resnet18.opt.use_local_pool_allocator = true;  // 使用本地内存池
            resnet18.opt.use_shader_pack8 = true;     // 使用shader pack8优化
            resnet18.opt.use_shader_local_memory = true;  // 使用本地显存

            if (!(resnet18.load_param("model_param/ncnn/resnet18.param") || 
                resnet18.load_model("model_param/ncnn/resnet18.bin"))) {
                manager.setLoaded(true);
                init_success = true;
            }
            // // 输出当前配置
            // printf("Using Vulkan: %d\n", resnet18.opt.use_vulkan_compute);
            printf("Thread Count: %d\n", resnet18.opt.num_threads);
            
        }
    });
    return init_success;
}

static PerformanceMetrics detect_resnet18_with_metrics(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    auto& manager = ModelManager::getInstance();
    auto& resnet18 = manager.getModel();
    PerformanceMetrics metrics;
    try {
        size_t start_memory = getCurrentMemoryUsage();
        auto start_preprocess = std::chrono::high_resolution_clock::now();

        if(!initModel()) {
            fprintf(stderr, "Model initialization failed\n");
            return metrics;
        }


        //opencv读取图片是BGR格式，我们需要转换为RGB格式
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
        
        //图像归一标准化，以R通道为例（x/225-0.485）/0.229，化简后可以得到下面的式子
        //需要注意的式substract_mean_normalize里的方差其实是方差的倒数，这样在算的时候就可以将除法转换为乘法计算
        //所以norm_vals里用的是1除
        const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
        const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto end_preprocess = std::chrono::high_resolution_clock::now();
        metrics.preprocessing_time_ms = std::chrono::duration<double, std::milli>(
            end_preprocess - start_preprocess).count();
        // 推理
        auto start_inference = std::chrono::high_resolution_clock::now();
        ncnn::Extractor ex = resnet18.create_extractor();
        
        //把图像数据放入data这个blob里
        ex.input("data", in);

        ncnn::Mat out;
        //提取出推理结果，推理结果存放在resnetv22_dense0_fwd这个blob里
        ex.extract("resnetv22_dense0_fwd", out);

        auto end_inference = std::chrono::high_resolution_clock::now();
        metrics.inference_time_ms = std::chrono::duration<double, std::milli>(
            end_inference - start_inference).count();

        // 后处理
        auto start_postprocess = std::chrono::high_resolution_clock::now();

        cls_scores.resize(out.w);
        for (int j = 0; j < out.w; j++)
        {
            cls_scores[j] = out[j];
        }

        auto end_postprocess = std::chrono::high_resolution_clock::now();
        metrics.postprocessing_time_ms = std::chrono::duration<double, std::milli>(
            end_postprocess - start_postprocess).count();
        // 记录内存使用
        metrics.memory_usage_kb = getCurrentMemoryUsage() - start_memory;
        return metrics;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "Error in detection: %s\n", e.what());
        return metrics;
    }
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        printf("Class: %d, Score: %.6f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
            return -1;
        }

        const char* imagepath = argv[1];
        if (!initModel()) {
            fprintf(stderr, "Model initialization failed\n");
            return -1;
        }
        //使用opencv读取图片
        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }

        // std::vector<float> cls_scores;
        // detect_resnet18(m, cls_scores);
        std::vector<float> cls_scores;
        // 在度量收集之前记录基准内存
        size_t base_memory = getCurrentMemoryUsage();
        const int warmup_runs = 1;  // 添加热身运行
        const int num_runs = 10; // 运行10次取平均
        std::vector<PerformanceMetrics> metrics_vec;
        // // 热身运行
        // for(int i = 0; i < warmup_runs; i++) {
        //     detect_resnet18_with_metrics(m, cls_scores);
        // }
        // 正式运行收集数据
        metrics_vec.reserve(num_runs);
        for(int i = 0; i < num_runs; i++) {
            metrics_vec.push_back(detect_resnet18_with_metrics(m, cls_scores));
        }

        // 计算统计指标
        double avg_inference_time = 0.0, max_inference_time = 0.0, min_inference_time = DBL_MAX;
        double avg_preprocess_time = 0.0, max_preprocess_time = 0.0, min_preprocess_time = DBL_MAX;
        double avg_postprocess_time = 0.0, max_postprocess_time = 0.0, min_postprocess_time = DBL_MAX;
        // 添加总时间统计变量
        double min_total_time = DBL_MAX;
        double max_total_time = 0.0;
        double total_time = 0.0;
        
        // 修改内存统计部分
        size_t total_memory = 0;
        size_t peak_memory = 0;
        for(const auto& metrics : metrics_vec) {
            double iter_total_time = metrics.preprocessing_time_ms + 
                               metrics.inference_time_ms + 
                               metrics.postprocessing_time_ms;
            min_total_time = std::min(min_total_time, iter_total_time);
            max_total_time = std::max(max_total_time, iter_total_time);
            total_time += iter_total_time;
            // 推理时间统计
            avg_inference_time += metrics.inference_time_ms;
            max_inference_time = std::max(max_inference_time, metrics.inference_time_ms);
            min_inference_time = std::min(min_inference_time, metrics.inference_time_ms);
            
            // 预处理时间统计
            avg_preprocess_time += metrics.preprocessing_time_ms;
            max_preprocess_time = std::max(max_preprocess_time, metrics.preprocessing_time_ms);
            min_preprocess_time = std::min(min_preprocess_time, metrics.preprocessing_time_ms);
            
            // 后处理时间统计
            avg_postprocess_time += metrics.postprocessing_time_ms;
            max_postprocess_time = std::max(max_postprocess_time, metrics.postprocessing_time_ms);
            min_postprocess_time = std::min(min_postprocess_time, metrics.postprocessing_time_ms);
            peak_memory = std::max(peak_memory, metrics.memory_usage_kb);
            total_memory += metrics.memory_usage_kb;
        }
        
        avg_inference_time /= num_runs;
        avg_preprocess_time /= num_runs;
        avg_postprocess_time /= num_runs;
        total_time /= num_runs; 
        size_t avg_memory = total_memory / num_runs;
        // 输出性能报告
        std::cout << "\n=== Performance Report (" << num_runs << " runs) ===" << std::endl;
        printf("Min/Max/Avg Preprocessing Time: %.2f/%.2f/%.2f ms\n", 
            min_preprocess_time, max_preprocess_time, avg_preprocess_time);
        printf("Min/Max/Avg Inference Time: %.2f/%.2f/%.2f ms\n",
            min_inference_time, max_inference_time, avg_inference_time);
        printf("Min/Max/Avg Postprocessing Time: %.2f/%.2f/%.2f ms\n",
            min_postprocess_time, max_postprocess_time, avg_postprocess_time);

        printf("Memory Usage: Base=%zu KB, Peak=%zu KB, Growth=%zu KB\n",
           base_memory, base_memory + peak_memory, peak_memory);
        printf("Min/Max/Avg FPS: %.2f/%.2f/%.2f\n",
                1000.0/max_total_time, 1000.0/min_total_time, 1000.0/total_time);

        printf("\nTop %d Results:\n", 3);
        print_topk(cls_scores, 3);
        // 修改清理顺序
        {
            auto& manager = ModelManager::getInstance();
            auto& model = manager.getModel();
            model.opt.use_vulkan_compute = false;
            model.clear();
        }
        return 0;
    }
    catch(const std::exception& e)
    {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }
    
    
}
