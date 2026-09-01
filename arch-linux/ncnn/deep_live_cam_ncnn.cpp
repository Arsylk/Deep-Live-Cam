// Minimal ncnn/Vulkan bridge used by the Python face-swapper adapter.
// The public ABI intentionally deals only in contiguous float32 tensors.

#include <cmath>
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

#include <ncnn/gpu.h>
#include <ncnn/net.h>

#if defined(_WIN32)
#define DLC_NCNN_EXPORT __declspec(dllexport)
#else
#define DLC_NCNN_EXPORT __attribute__((visibility("default")))
#endif

namespace
{

constexpr int kWidth = 128;
constexpr int kHeight = 128;
constexpr int kChannels = 3;
constexpr int kEmbeddingSize = 512;
constexpr size_t kPlaneSize = static_cast<size_t>(kWidth) * kHeight;
constexpr int kNativeWidth = 256;
constexpr int kNativeHeight = 256;
constexpr int kNativeChannels = 3;
constexpr int kNativeStyleSize = 2016;
constexpr size_t kNativePlaneSize =
    static_cast<size_t>(kNativeWidth) * kNativeHeight;

thread_local std::string last_error;
std::mutex instance_mutex;
unsigned int instance_users = 0;

struct Swapper
{
    ncnn::Net net;
    std::mutex run_mutex;
    std::string gpu_name;
};

struct Native256Swapper
{
    ncnn::Net conditioner;
    ncnn::Net generator;
    std::mutex run_mutex;
    std::string gpu_name;
    std::array<float, kEmbeddingSize> cached_identity{};
    ncnn::Mat cached_style;
    bool has_cached_style = false;
};

void set_error(const std::string& message)
{
    last_error = message;
}

void release_gpu_instance()
{
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (instance_users == 0)
        return;
    --instance_users;
    if (instance_users == 0)
        ncnn::destroy_gpu_instance();
}

bool acquire_gpu_instance()
{
    std::lock_guard<std::mutex> lock(instance_mutex);
    if (instance_users == 0)
    {
        const int status = ncnn::create_gpu_instance();
        if (status != 0)
        {
            set_error("ncnn could not initialize Vulkan (status " +
                      std::to_string(status) + ")");
            return false;
        }
        if (ncnn::get_gpu_count() < 1)
        {
            ncnn::destroy_gpu_instance();
            set_error("ncnn initialized Vulkan but found no compute device");
            return false;
        }
    }
    ++instance_users;
    return true;
}

void configure_net(ncnn::Net& net, int device_index, bool fp16_storage,
                   bool light_mode)
{
    ncnn::Option& option = net.opt;
    option.num_threads = 8;
    option.lightmode = light_mode;
    option.use_vulkan_compute = true;
    option.vulkan_device_index = device_index;
    option.use_fp16_packed = fp16_storage;
    option.use_fp16_storage = fp16_storage;
    // Polaris supports 16-bit storage but not shader float16 arithmetic.
    // Storage is controlled by the parity-qualified bundle manifest; arithmetic
    // remains disabled so the same bridge is safe on older AMD hardware.
    option.use_fp16_arithmetic = false;
    option.use_bf16_storage = false;
}

bool finite_tensor(const ncnn::Mat& tensor)
{
    if (tensor.dims != 3 || tensor.elempack != 1 ||
        tensor.elemsize != sizeof(float))
        return false;
    // ncnn aligns channel strides. ``total()`` includes that uninitialized
    // padding, so inspect only the logical w*h values in each plane.
    const size_t plane_size = static_cast<size_t>(tensor.w) * tensor.h;
    for (int channel = 0; channel < tensor.c; ++channel)
    {
        const float* values = tensor.channel(channel);
        for (size_t index = 0; index < plane_size; ++index)
        {
            if (!std::isfinite(values[index]))
                return false;
        }
    }
    return true;
}

bool finite_array(const float* values, size_t count)
{
    for (size_t index = 0; index < count; ++index)
    {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

int condition_native_identity(Native256Swapper& swapper,
                              const float* mapped_identity)
{
    if (swapper.has_cached_style &&
        std::memcmp(swapper.cached_identity.data(), mapped_identity,
                    kEmbeddingSize * sizeof(float)) == 0)
    {
        return 0;
    }

    ncnn::Mat identity(1, 1, kEmbeddingSize, sizeof(float));
    if (identity.empty())
    {
        set_error("could not allocate native-256 identity tensor");
        return -20;
    }
    for (int channel = 0; channel < kEmbeddingSize; ++channel)
        identity.channel(channel)[0] = mapped_identity[channel];

    ncnn::Extractor extractor = swapper.conditioner.create_extractor();
    int status = extractor.input("in0", identity);
    ncnn::Mat style;
    if (status == 0)
        status = extractor.extract("out0", style);
    if (status != 0)
    {
        set_error("native-256 conditioner failed (status " +
                  std::to_string(status) + ")");
        return status;
    }
    if (style.dims != 3 || style.w != 1 || style.h != 1 ||
        style.c != kNativeStyleSize || !finite_tensor(style))
    {
        set_error("native-256 conditioner returned an invalid style tensor " +
                  std::to_string(style.dims) + "D " +
                  std::to_string(style.w) + "x" +
                  std::to_string(style.h) + "x" +
                  std::to_string(style.c) + " pack=" +
                  std::to_string(style.elempack) + " elemsize=" +
                  std::to_string(style.elemsize));
        return -21;
    }

    // Keep a CPU-owned copy beyond the extractor lifetime. It is uploaded by
    // each generator extractor but the conditioner itself is skipped until the
    // exact mapped source identity changes.
    swapper.cached_style = style.clone();
    if (swapper.cached_style.empty())
    {
        set_error("could not cache the native-256 style tensor");
        return -22;
    }
    std::memcpy(swapper.cached_identity.data(), mapped_identity,
                kEmbeddingSize * sizeof(float));
    swapper.has_cached_style = true;
    return 0;
}

} // namespace

extern "C"
{

DLC_NCNN_EXPORT const char* dlc_ncnn_last_error()
{
    return last_error.c_str();
}

DLC_NCNN_EXPORT int dlc_ncnn_abi_version()
{
    return 2;
}

DLC_NCNN_EXPORT void* dlc_ncnn_swapper_create(const char* param_path,
                                               const char* model_path,
                                               int device_index)
{
    last_error.clear();
    if (!param_path || !model_path)
    {
        set_error("model paths must not be null");
        return nullptr;
    }
    if (!acquire_gpu_instance())
        return nullptr;

    if (device_index < 0 || device_index >= ncnn::get_gpu_count())
    {
        set_error("requested Vulkan device index is unavailable");
        release_gpu_instance();
        return nullptr;
    }

    Swapper* swapper = new (std::nothrow) Swapper();
    if (!swapper)
    {
        set_error("could not allocate the ncnn swapper context");
        release_gpu_instance();
        return nullptr;
    }

    swapper->gpu_name = ncnn::get_gpu_info(device_index).device_name();
    // INSwapper squares large pre-normalization activations, so this legacy
    // graph deliberately remains FP32 even where the device supports storage.
    configure_net(swapper->net, device_index, false, true);

    int status = swapper->net.load_param(param_path);
    if (status == 0)
        status = swapper->net.load_model(model_path);
    if (status != 0)
    {
        set_error("ncnn failed to load the swapper (status " +
                  std::to_string(status) + ")");
        delete swapper;
        release_gpu_instance();
        return nullptr;
    }

    return swapper;
}

DLC_NCNN_EXPORT void dlc_ncnn_swapper_destroy(void* context)
{
    if (!context)
        return;
    delete static_cast<Swapper*>(context);
    release_gpu_instance();
}

DLC_NCNN_EXPORT const char* dlc_ncnn_swapper_device_name(void* context)
{
    if (!context)
        return "";
    return static_cast<Swapper*>(context)->gpu_name.c_str();
}

DLC_NCNN_EXPORT int dlc_ncnn_swapper_run(void* context,
                                         const float* target,
                                         const float* source,
                                         float* output)
{
    last_error.clear();
    if (!context || !target || !source || !output)
    {
        set_error("context and tensor pointers must not be null");
        return -1;
    }

    Swapper& swapper = *static_cast<Swapper*>(context);
    std::lock_guard<std::mutex> lock(swapper.run_mutex);

    ncnn::Mat target_mat(kWidth, kHeight, kChannels, sizeof(float));
    ncnn::Mat source_mat(kEmbeddingSize, sizeof(float));
    if (target_mat.empty() || source_mat.empty())
    {
        set_error("could not allocate ncnn input tensors");
        return -2;
    }
    for (int channel = 0; channel < kChannels; ++channel)
    {
        std::memcpy(target_mat.channel(channel),
                    target + static_cast<size_t>(channel) * kPlaneSize,
                    kPlaneSize * sizeof(float));
    }
    std::memcpy(source_mat.data, source, kEmbeddingSize * sizeof(float));

    ncnn::Extractor extractor = swapper.net.create_extractor();
    int status = extractor.input("in0", target_mat);
    if (status == 0)
        status = extractor.input("in1", source_mat);

    ncnn::Mat result;
    if (status == 0)
        status = extractor.extract("out0", result);
    if (status != 0)
    {
        set_error("ncnn inference failed (status " + std::to_string(status) + ")");
        return status;
    }
    if (result.dims != 3 || result.w != kWidth || result.h != kHeight ||
        result.c != kChannels || result.elempack != 1 ||
        result.elemsize != sizeof(float))
    {
        set_error("ncnn returned an unexpected output tensor layout");
        return -3;
    }

    for (int channel = 0; channel < kChannels; ++channel)
    {
        const float* plane = result.channel(channel);
        for (size_t index = 0; index < kPlaneSize; ++index)
        {
            if (!std::isfinite(plane[index]))
            {
                set_error("ncnn returned a non-finite output value");
                return -4;
            }
        }
        std::memcpy(output + static_cast<size_t>(channel) * kPlaneSize,
                    plane, kPlaneSize * sizeof(float));
    }
    return 0;
}

DLC_NCNN_EXPORT void* dlc_ncnn_native256_create(
    const char* conditioner_param_path, const char* conditioner_model_path,
    const char* generator_param_path, const char* generator_model_path,
    int device_index, int fp16_storage)
{
    last_error.clear();
    if (!conditioner_param_path || !conditioner_model_path ||
        !generator_param_path || !generator_model_path)
    {
        set_error("native-256 model paths must not be null");
        return nullptr;
    }
    if (fp16_storage != 0 && fp16_storage != 1)
    {
        set_error("native-256 fp16_storage must be zero or one");
        return nullptr;
    }
    if (!acquire_gpu_instance())
        return nullptr;
    if (device_index < 0 || device_index >= ncnn::get_gpu_count())
    {
        set_error("requested Vulkan device index is unavailable");
        release_gpu_instance();
        return nullptr;
    }

    Native256Swapper* swapper = new (std::nothrow) Native256Swapper();
    if (!swapper)
    {
        set_error("could not allocate the native-256 ncnn context");
        release_gpu_instance();
        return nullptr;
    }
    swapper->gpu_name = ncnn::get_gpu_info(device_index).device_name();
    // Two outputs share generator intermediates, so light mode must remain off
    // until both candidate and alpha have been extracted.
    configure_net(swapper->conditioner, device_index, fp16_storage != 0, true);
    configure_net(swapper->generator, device_index, fp16_storage != 0, false);

    int status = swapper->conditioner.load_param(conditioner_param_path);
    if (status == 0)
        status = swapper->conditioner.load_model(conditioner_model_path);
    if (status == 0)
        status = swapper->generator.load_param(generator_param_path);
    if (status == 0)
        status = swapper->generator.load_model(generator_model_path);
    if (status != 0)
    {
        set_error("ncnn failed to load the native-256 bundle (status " +
                  std::to_string(status) + ")");
        delete swapper;
        release_gpu_instance();
        return nullptr;
    }
    return swapper;
}

DLC_NCNN_EXPORT void dlc_ncnn_native256_destroy(void* context)
{
    if (!context)
        return;
    delete static_cast<Native256Swapper*>(context);
    release_gpu_instance();
}

DLC_NCNN_EXPORT const char* dlc_ncnn_native256_device_name(void* context)
{
    if (!context)
        return "";
    return static_cast<Native256Swapper*>(context)->gpu_name.c_str();
}

DLC_NCNN_EXPORT void dlc_ncnn_native256_clear_style(void* context)
{
    if (!context)
        return;
    Native256Swapper& swapper = *static_cast<Native256Swapper*>(context);
    std::lock_guard<std::mutex> lock(swapper.run_mutex);
    swapper.cached_style.release();
    swapper.cached_identity.fill(0.0f);
    swapper.has_cached_style = false;
}

DLC_NCNN_EXPORT int dlc_ncnn_native256_run(void* context,
                                           const float* target,
                                           const float* mapped_identity,
                                           float* candidate_output,
                                           float* alpha_output)
{
    last_error.clear();
    if (!context || !target || !mapped_identity || !candidate_output ||
        !alpha_output)
    {
        set_error("native-256 context and tensor pointers must not be null");
        return -10;
    }
    if (!finite_array(target, kNativePlaneSize * kNativeChannels) ||
        !finite_array(mapped_identity, kEmbeddingSize))
    {
        set_error("native-256 inputs contain a non-finite value");
        return -11;
    }

    Native256Swapper& swapper = *static_cast<Native256Swapper*>(context);
    std::lock_guard<std::mutex> lock(swapper.run_mutex);
    int status = condition_native_identity(swapper, mapped_identity);
    if (status != 0)
        return status;

    ncnn::Mat target_mat(kNativeWidth, kNativeHeight, kNativeChannels,
                         sizeof(float));
    if (target_mat.empty())
    {
        set_error("could not allocate native-256 target tensor");
        return -12;
    }
    for (int channel = 0; channel < kNativeChannels; ++channel)
    {
        std::memcpy(target_mat.channel(channel),
                    target + static_cast<size_t>(channel) * kNativePlaneSize,
                    kNativePlaneSize * sizeof(float));
    }

    ncnn::Extractor extractor = swapper.generator.create_extractor();
    status = extractor.input("in0", target_mat);
    if (status == 0)
        status = extractor.input("in1", swapper.cached_style);
    ncnn::Mat candidate;
    ncnn::Mat alpha;
    if (status == 0)
        status = extractor.extract("out0", candidate);
    if (status == 0)
        status = extractor.extract("out1", alpha);
    if (status != 0)
    {
        set_error("native-256 generator failed (status " +
                  std::to_string(status) + ")");
        return status;
    }
    if (candidate.dims != 3 || candidate.w != kNativeWidth ||
        candidate.h != kNativeHeight || candidate.c != kNativeChannels ||
        !finite_tensor(candidate))
    {
        set_error("native-256 generator returned an invalid candidate tensor " +
                  std::to_string(candidate.dims) + "D " +
                  std::to_string(candidate.w) + "x" +
                  std::to_string(candidate.h) + "x" +
                  std::to_string(candidate.c) + " pack=" +
                  std::to_string(candidate.elempack) + " elemsize=" +
                  std::to_string(candidate.elemsize));
        return -13;
    }
    if (alpha.dims != 3 || alpha.w != kNativeWidth ||
        alpha.h != kNativeHeight || alpha.c != 1 || !finite_tensor(alpha))
    {
        set_error("native-256 generator returned an invalid alpha tensor " +
                  std::to_string(alpha.dims) + "D " +
                  std::to_string(alpha.w) + "x" +
                  std::to_string(alpha.h) + "x" +
                  std::to_string(alpha.c) + " pack=" +
                  std::to_string(alpha.elempack) + " elemsize=" +
                  std::to_string(alpha.elemsize));
        return -14;
    }

    for (int channel = 0; channel < kNativeChannels; ++channel)
    {
        std::memcpy(candidate_output +
                        static_cast<size_t>(channel) * kNativePlaneSize,
                    candidate.channel(channel),
                    kNativePlaneSize * sizeof(float));
    }
    std::memcpy(alpha_output, alpha.channel(0),
                kNativePlaneSize * sizeof(float));
    return 0;
}

} // extern "C"
