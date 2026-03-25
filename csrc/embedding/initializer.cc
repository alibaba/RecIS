#include "embedding/initializer.h"

#include <cmath>
#include <string>
#include <utility>

#include "ATen/Dispatch.h"
#include "ATen/core/Generator.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/fill.h"
#include "ATen/ops/zeros.h"
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Logging.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/irange.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/init.h"
#include "torch/types.h"
namespace recis {
namespace embedding {
namespace {

class EmptyGenerator : public Generator {
 public:
  EmptyGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype)
      : Generator(shape, dtype) {}
  void Initialize(torch::Tensor ret) override {}
  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    torch::Tensor ret = torch::empty(shape_used, option_);
    return ret;
  }
};
class ConstantGenerator : public Generator {
 public:
  ConstantGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype,
                    double init_val)
      : Generator(shape, dtype), init_val_(init_val) {}

  void Initialize(torch::Tensor ret) override {
    AT_DISPATCH_ALL_TYPES_AND2(torch::kHalf, torch::kBFloat16,
                               option_.dtype().toScalarType(),
                               "##ConstantGenerator##", [this, &ret]() {
                                 scalar_t init_val = (scalar_t)init_val_;
                                 torch::fill_(ret, init_val);
                               });
  }
  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    torch::Tensor ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float init_val_;
};

class UniformGenerator : public Generator {
 public:
  UniformGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype,
                   double a, double b,
                   c10::optional<torch::Generator> generator)
      : Generator(shape, dtype), a_(a), b_(b), generator_(generator) {};

  void Initialize(at::Tensor ret) override { ret.uniform_(a_, b_, generator_); }

  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    auto ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float a_;
  float b_;
  c10::optional<torch::Generator> generator_;
};

class NormalGenerator : public Generator {
 public:
  NormalGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype,
                  double mean, double std,
                  c10::optional<torch::Generator> generator)
      : Generator(shape, dtype), mean_(mean), std_(std), generator_(generator) {
        };
  void Initialize(at::Tensor ret) override {
    ret.normal_(mean_, std_, generator_);
  }
  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    auto ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float mean_;
  float std_;
  c10::optional<torch::Generator> generator_;
};

class XavierUniformGenerator : public Generator {
 public:
  XavierUniformGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype,
                         double gain, c10::optional<torch::Generator> generator)
      : Generator(shape, dtype), gain_(gain), generator_(generator) {}

  void Initialize(at::Tensor ret) override {
    torch::nn::init::xavier_uniform_(ret, gain_);
  }

  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    auto ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float gain_;
  c10::optional<torch::Generator> generator_;
};

class XavierNormalGenerator : public Generator {
 public:
  XavierNormalGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype,
                        double gain, c10::optional<torch::Generator> generator)
      : Generator(shape, dtype), gain_(gain), generator_(generator) {}

  void Initialize(at::Tensor ret) override {
    torch::nn::init::xavier_normal_(ret, gain_);
  }

  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    auto ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float gain_;
  c10::optional<torch::Generator> generator_;
};

class KaimingUnifomGenerator : public Generator {
 public:
  KaimingUnifomGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype,
                         double a, torch::nn::init::FanModeType fan_mode,
                         torch::nn::init::NonlinearityType non_lineariry,
                         c10::optional<torch::Generator> generator)
      : Generator(shape, dtype),
        a_(a),
        mode_(fan_mode),
        nonlinearity_(non_lineariry) {}
  void Initialize(at::Tensor ret) override {
    torch::nn::init::kaiming_uniform_(ret, a_, mode_, nonlinearity_);
  }
  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    auto ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float a_;
  torch::nn::init::FanModeType mode_;
  torch::nn::init::NonlinearityType nonlinearity_;
  c10::optional<torch::Generator> generator_;
};

class KaimingNormalGenerator : public Generator {
 public:
  KaimingNormalGenerator(const std::vector<int64_t> &shape, torch::Dtype dtype,
                         double a, torch::nn::init::FanModeType fan_mode,
                         torch::nn::init::NonlinearityType non_lineariry,
                         c10::optional<torch::Generator> generator)
      : Generator(shape, dtype),
        a_(a),
        mode_(fan_mode),
        nonlinearity_(non_lineariry) {}

  void Initialize(at::Tensor ret) override {
    torch::nn::init::kaiming_normal_(ret, a_, mode_, nonlinearity_);
  }
  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    auto ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float a_;
  torch::nn::init::FanModeType mode_;
  torch::nn::init::NonlinearityType nonlinearity_;
  c10::optional<torch::Generator> generator_;
};

class TruncNormalGenerator : public Generator {
 public:
  TruncNormalGenerator(const std::vector<int64_t> &shape, torch::Dtype type,
                       double mean, double std, double a, double b,
                       at::optional<torch::Generator> generator)
      : Generator(shape, type),
        mean_(mean),
        std_(std),
        a_(a),
        b_(b),
        generator_(generator) {}
  void Initialize(at::Tensor ret) override {
    if (mean_ < (a_ - 2 * std_) || mean_ > (b_ + 2 * std_)) {
      LOG(WARNING)
          << "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
             "The distribution of values may be incorrect.";
    }
    float l = normal_cdf((a_ - mean_) / std_);
    float u = normal_cdf((b_ - mean_) / std_);
    ret.uniform_(2 * l - 1, 2 * u - 1, generator_);
    ret.erfinv_();
    ret.mul_(std_ * std::sqrt(2.0));
    ret.add_(mean_);
    ret.clamp_(a_, b_);
  }
  torch::Tensor Generate(const std::vector<int64_t> &shape = {}) override {
    const std::vector<int64_t> &shape_used = shape.empty() ? shape_ : shape;
    auto ret = torch::empty(shape_used, option_);
    Initialize(ret);
    return ret;
  }

 private:
  float normal_cdf(float x) {
    return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
  }
  float mean_;
  float std_;
  float a_;
  float b_;
  at::optional<torch::Generator> generator_;
};

torch::nn::init::NonlinearityType StringToNolinearity(const std::string &type) {
  if (type == "liner") {
    return torch::kLinear;
  } else if (type == "conv_1d") {
    return torch::kConv1D;
  } else if (type == "conv_2d") {
    return torch::kConv2D;
  } else if (type == "conv_3d") {
    return torch::kConv3D;
  } else if (type == "conv_transpose_1d") {
    return torch::kConvTranspose1D;

  } else if (type == "conv_transpose_2d") {
    return torch::kConvTranspose2D;

  } else if (type == "conv_transpose_3d") {
    return torch::kConvTranspose3D;

  } else if (type == "sigmod") {
    return torch::kSigmoid;
  } else if (type == "tanh") {
    return torch::kTanh;
  } else if (type == "relu") {
    return torch::kReLU;
  } else if (type == "leaky_relu") {
    return torch::kLeakyReLU;
  } else {
    TORCH_CHECK(false, type, " not in  [liner, conv_1d, conv_2d, cond3d,",
                " conv_transpose_1d, conv_transpose_2d, conv_transpose_3d,",
                "sigmod, tanh, relu, leaky_relu"
                "]")
  }
}
torch::nn::init::FanModeType StringToFanModeType(const std::string &type) {
  if (type == "fan_in") {
    return torch::kFanIn;
  } else if (type == "fan_out") {
    return torch::kFanOut;
  } else {
    TORCH_CHECK(false, type, " not in [fan_in, fan_out]")
  }
}
}  // namespace

Generator::Generator(const std::vector<int64_t> &shape, torch::Dtype dtype)
    : shape_(shape) {
  option_ = torch::TensorOptions().device(torch::kCPU).dtype(dtype);
}

void Generator::set_device(torch::Device device) {
  option_ = torch::TensorOptions(option_).device(device);
}

torch::Tensor Generator::DoGenerator(torch::intrusive_ptr<Generator> genrate) {
  return genrate->Generate();
}

torch::intrusive_ptr<Generator> MakeEmptyGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype) {
  return at::make_intrusive<EmptyGenerator>(shape, dtype);
}

torch::intrusive_ptr<Generator> MakeConstantGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double init_val) {
  return c10::make_intrusive<ConstantGenerator>(shape, dtype, init_val);
}

torch::intrusive_ptr<Generator> MakeUniformGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double a, double b,
    c10::optional<torch::Generator> generator) {
  return c10::make_intrusive<UniformGenerator>(shape, dtype, a, b, generator);
}

torch::intrusive_ptr<Generator> MakeNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double mean,
    double std, c10::optional<torch::Generator> generator) {
  return c10::make_intrusive<NormalGenerator>(shape, dtype, mean, std,
                                              generator);
}

torch::intrusive_ptr<Generator> MakeXavierUniFormGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double gain,
    c10::optional<torch::Generator> generator) {
  return c10::make_intrusive<XavierUniformGenerator>(shape, dtype, gain,
                                                     generator);
}

torch::intrusive_ptr<Generator> MakeXavierNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double gain,
    c10::optional<torch::Generator> generator) {
  return c10::make_intrusive<XavierNormalGenerator>(shape, dtype, gain,
                                                    generator);
}

torch::intrusive_ptr<Generator> MakeKaimingUniformGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double a,
    const std::string &mode_s, const std::string &nonlinearity_s,
    c10::optional<torch::Generator> generator) {
  auto mode = StringToFanModeType(mode_s);
  auto nonlinearity = StringToNolinearity(nonlinearity_s);
  return c10::make_intrusive<KaimingUnifomGenerator>(shape, dtype, a, mode,
                                                     nonlinearity, generator);
}

torch::intrusive_ptr<Generator> MakeKaimingNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype dtype, double a,
    const std::string &mode_s, const std::string &nonlinearity_s,
    c10::optional<torch::Generator> generator) {
  auto mode = StringToFanModeType(mode_s);
  auto nonlinearity = StringToNolinearity(nonlinearity_s);
  return c10::make_intrusive<KaimingNormalGenerator>(shape, dtype, a, mode,
                                                     nonlinearity, generator);
}

torch::intrusive_ptr<Generator> MakeTruncNormalGenerator(
    const std::vector<int64_t> &shape, torch::Dtype type, double mean,
    double std, double a, double b, at::optional<torch::Generator> generator) {
  return c10::make_intrusive<TruncNormalGenerator>(shape, type, mean, std, a, b,
                                                   generator);
}
}  // namespace embedding
}  // namespace recis
