// Isolated raw-tensor benchmark for the converted INSwapper ncnn model.
//
// This deliberately excludes camera, alignment, and post-processing work so
// that CPU and Vulkan backends can be compared with identical deterministic
// tensors. It is not linked into the application.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <ncnn/gpu.h>
#include <ncnn/net.h>

namespace
{

using Clock = std::chrono::steady_clock;

struct Options
{
    std::string param_path;
    std::string bin_path;
    std::string backend;
    int runs = 20;
    int warmups = 10;
    std::string output_blob = "out0";
    std::string output_path;
};

void usage(const char* program)
{
    std::fprintf(stderr,
                 "usage: %s MODEL.param MODEL.bin "
                 "{cpu|vulkan-fp32|vulkan-fp16-storage} [runs] [output.f32]\n",
                 program);
}

bool parse_options(int argc, char** argv, Options& options)
{
    if (argc < 4 || argc > 6)
        return false;

    options.param_path = argv[1];
    options.bin_path = argv[2];
    options.backend = argv[3];
    if (options.backend != "cpu" && options.backend != "vulkan-fp32" &&
        options.backend != "vulkan-fp16-storage")
        return false;

    if (argc >= 5)
    {
        try
        {
            options.runs = std::stoi(argv[4]);
        }
        catch (...)
        {
            return false;
        }
        if (options.runs < 1)
            return false;
    }
    if (argc == 6)
        options.output_path = argv[5];

    if (const char* value = std::getenv("NCNN_BENCH_WARMUPS"))
    {
        try
        {
            options.warmups = std::stoi(value);
        }
        catch (...)
        {
            return false;
        }
        if (options.warmups < 0)
            return false;
    }
    if (const char* value = std::getenv("NCNN_BENCH_OUTPUT_BLOB"))
    {
        if (*value == '\0')
            return false;
        options.output_blob = value;
    }
    return true;
}

void fill_inputs(ncnn::Mat& target, ncnn::Mat& source)
{
    // Mat storage is planar CHW, matching ONNX's contiguous NCHW tensor.
    float* target_values = static_cast<float*>(target.data);
    for (size_t i = 0; i < target.total(); ++i)
    {
        target_values[i] = 0.5f + 0.25f * std::sin(static_cast<float>(i) * 0.013f) +
                           0.125f * std::cos(static_cast<float>(i) * 0.007f);
    }

    float* source_values = static_cast<float*>(source.data);
    double squared_norm = 0.0;
    for (size_t i = 0; i < source.total(); ++i)
    {
        const float value = std::sin(static_cast<float>(i + 1) * 0.071f) +
                            0.5f * std::cos(static_cast<float>(i + 3) * 0.037f);
        source_values[i] = value;
        squared_norm += static_cast<double>(value) * value;
    }
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (size_t i = 0; i < source.total(); ++i)
        source_values[i] *= inverse_norm;
}

int run_once(ncnn::Net& net, const ncnn::Mat& target, const ncnn::Mat& source,
             const char* output_blob, ncnn::Mat& output)
{
    ncnn::Extractor extractor = net.create_extractor();
    int status = extractor.input("in0", target);
    if (status != 0)
        return status;
    status = extractor.input("in1", source);
    if (status != 0)
        return status;
    return extractor.extract(output_blob, output);
}

double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

bool write_output(const std::string& path, const ncnn::Mat& output)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    stream.write(reinterpret_cast<const char*>(output.data),
                 static_cast<std::streamsize>(output.total() * sizeof(float)));
    return stream.good();
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_options(argc, argv, options))
    {
        usage(argv[0]);
        return 2;
    }

    const bool use_vulkan = options.backend != "cpu";
    if (use_vulkan)
    {
        const int status = ncnn::create_gpu_instance();
        if (status != 0)
        {
            std::fprintf(stderr, "failed to initialize Vulkan: %d\n", status);
            return 3;
        }
        const int gpu_count = ncnn::get_gpu_count();
        if (gpu_count < 1)
        {
            std::fprintf(stderr, "Vulkan initialized but ncnn found no GPU\n");
            ncnn::destroy_gpu_instance();
            return 3;
        }
        const ncnn::GpuInfo& gpu = ncnn::get_gpu_info(0);
        std::printf("gpu=%s driver=%s api=%u.%u.%u fp16_storage=%s "
                    "fp16_arithmetic=%s\n",
                    gpu.device_name(), gpu.driver_name(),
                    VK_VERSION_MAJOR(gpu.api_version()),
                    VK_VERSION_MINOR(gpu.api_version()),
                    VK_VERSION_PATCH(gpu.api_version()),
                    gpu.support_fp16_storage() ? "yes" : "no",
                    gpu.support_fp16_arithmetic() ? "yes" : "no");
    }

    int result = 0;
    {
        ncnn::Net net;
        net.opt.num_threads = 8;
        net.opt.use_vulkan_compute = use_vulkan;
        net.opt.vulkan_device_index = 0;
        net.opt.use_fp16_packed = options.backend == "vulkan-fp16-storage";
        net.opt.use_fp16_storage = options.backend == "vulkan-fp16-storage";
        // Polaris exposes 16-bit storage but not shader float16 arithmetic.
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_bf16_storage = false;

        const auto load_start = Clock::now();
        int status = net.load_param(options.param_path.c_str());
        if (status == 0)
            status = net.load_model(options.bin_path.c_str());
        const double load_ms = std::chrono::duration<double, std::milli>(
                                   Clock::now() - load_start)
                                   .count();
        if (status != 0)
        {
            std::fprintf(stderr, "model load failed: %d\n", status);
            result = 4;
        }
        else
        {
            ncnn::Mat target(128, 128, 3, sizeof(float));
            ncnn::Mat source(512, sizeof(float));
            fill_inputs(target, source);

            ncnn::Mat output;
            // Vulkan creates pipelines lazily. Exclude ten warm-ups so shader
            // compilation does not masquerade as steady-state inference.
            for (int i = 0; i < options.warmups; ++i)
            {
                status = run_once(net, target, source,
                                  options.output_blob.c_str(), output);
                if (status != 0)
                    break;
            }

            std::vector<double> timings;
            timings.reserve(static_cast<size_t>(options.runs));
            for (int i = 0; status == 0 && i < options.runs; ++i)
            {
                const auto start = Clock::now();
                status = run_once(net, target, source,
                                  options.output_blob.c_str(), output);
                timings.push_back(std::chrono::duration<double, std::milli>(
                                      Clock::now() - start)
                                      .count());
            }

            if (status != 0)
            {
                std::fprintf(stderr, "inference failed: %d\n", status);
                result = 5;
            }
            else if (options.output_blob == "out0" &&
                     (output.w != 128 || output.h != 128 || output.c != 3 ||
                      output.elempack != 1 || output.elemsize != sizeof(float)))
            {
                std::fprintf(stderr,
                             "unexpected output: dims=%d w=%d h=%d c=%d "
                             "elemsize=%zu elempack=%d\n",
                             output.dims, output.w, output.h, output.c,
                             output.elemsize, output.elempack);
                result = 6;
            }
            else
            {
                if (output.elemsize / output.elempack != sizeof(float))
                {
                    std::fprintf(stderr,
                                 "diagnostic blob %s is not float32: "
                                 "elemsize=%zu elempack=%d\n",
                                 options.output_blob.c_str(), output.elemsize,
                                 output.elempack);
                    result = 6;
                }
                else
                {
                    const float* values = static_cast<const float*>(output.data);
                    const size_t count = output.total();
                    float minimum = values[0];
                    float maximum = values[0];
                    double sum = 0.0;
                    bool finite = true;
                    for (size_t i = 0; i < count; ++i)
                    {
                        minimum = std::min(minimum, values[i]);
                        maximum = std::max(maximum, values[i]);
                        sum += values[i];
                        finite = finite && std::isfinite(values[i]);
                    }

                    const double mean_ms = std::accumulate(timings.begin(), timings.end(), 0.0) /
                                           static_cast<double>(timings.size());
                    std::printf("backend=%s blob=%s dims=%d shape=%dx%dx%dx%d pack=%d "
                                "load_ms=%.3f warmups=%d runs=%d mean_ms=%.3f "
                                "p50_ms=%.3f p95_ms=%.3f min=%.7g max=%.7g "
                                "mean=%.7g finite=%s\n",
                                options.backend.c_str(), options.output_blob.c_str(),
                                output.dims, output.w, output.h, output.d,
                                output.c, output.elempack,
                                load_ms, options.warmups, options.runs,
                                mean_ms, percentile(timings, 0.50),
                                percentile(timings, 0.95), minimum, maximum,
                                sum / static_cast<double>(count), finite ? "yes" : "no");

                    if (!finite)
                        result = 7;
                    else if (!options.output_path.empty() &&
                             !write_output(options.output_path, output))
                    {
                        std::fprintf(stderr, "failed to write %s\n",
                                     options.output_path.c_str());
                        result = 8;
                    }
                }
            }
        }
    }

    if (use_vulkan)
        ncnn::destroy_gpu_instance();
    return result;
}
