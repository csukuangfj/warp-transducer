/**
 * @brief PyTorch Python Wrapper warp-rnnt
 *
 * Copyright      2021  Xiaomi Corp.  (authors: Fangjun Kuang)
 *
 */

#include "pybind11/pybind11.h"
#include "rnnt.h"
#include "torch/extension.h"

#ifdef RNNT_WITH_CUDA
#include "c10/cuda/CUDAStream.h"
#include "torch/cuda.h"
#endif

// See the comments in ComputeRnntLoss() below
// for the meaning of the arguments
static std::pair<torch::Tensor, torch::optional<torch::Tensor>>
CpuImpl(torch::Tensor log_probs, torch::Tensor targets,
        torch::Tensor log_prob_lengths, torch::Tensor target_lengths,
        int32_t blank, float fastemit_lambda) {
  // Check devices
  TORCH_CHECK(log_probs.device().is_cpu(), "log_probs.device() ",
              log_probs.device(), " is not CPU");

  TORCH_CHECK(log_prob_lengths.device().is_cpu(), "log_prob_lengths.device() ",
              log_prob_lengths.device(), " is not CPU");

  TORCH_CHECK(target_lengths.device().is_cpu(), "target_lengths.device() ",
              target_lengths.device(), " is not CPU");

  TORCH_CHECK(targets.device().is_cpu(), "targets.device() ", targets.device(),
              " is not CPU");

  // activations.shape is (N, T, U, C)
  int32_t N = log_probs.size(0);
  int32_t T = log_probs.size(1);
  int32_t U = log_probs.size(2);
  int32_t C = log_probs.size(3);

  rnntOptions options{};
  options.loc = RNNT_CPU;
  options.num_threads = 1;
  options.stream = nullptr;
  options.blank_label = blank;
  options.maxT = T;
  options.maxU = U;
  options.batch_first = true;
  options.fastemit_lambda = fastemit_lambda;

  torch::optional<torch::Tensor> grads;
  if (log_probs.requires_grad()) {
    grads = torch::empty_like(log_probs);
  }

  // costs is always on CPU
  torch::Tensor costs = torch::empty({N}, log_probs.dtype());

  size_t cpu_size_bytes = 0;

  if (log_probs.scalar_type() == torch::kFloat) {
    get_workspace_size(T, U, N,
                       /*gpu*/ false, &cpu_size_bytes, sizeof(float));

    uint8_t *cpu_workspace = new uint8_t[cpu_size_bytes];

    compute_rnnt_loss(log_probs.data_ptr<float>(),
                      grads ? grads->data_ptr<float>() : nullptr,
                      targets.data_ptr<int>(), target_lengths.data_ptr<int>(),
                      log_prob_lengths.data_ptr<int>(), C, N,
                      costs.data_ptr<float>(), cpu_workspace, options);

    delete[] cpu_workspace;
  } else {
    get_workspace_size(T, U, N,
                       /*gpu*/ false, &cpu_size_bytes, sizeof(double));
    uint8_t *cpu_workspace = new uint8_t[cpu_size_bytes];

    compute_rnnt_loss_fp64(
        log_probs.data_ptr<double>(),
        grads ? grads->data_ptr<double>() : nullptr, targets.data_ptr<int>(),
        target_lengths.data_ptr<int>(), log_prob_lengths.data_ptr<int>(), C, N,
        costs.data_ptr<double>(), cpu_workspace, options);
    delete[] cpu_workspace;
  }
  return {costs, grads};
}

// See the comments in ComputeRnntLoss() below
// for the meaning of the arguments
static std::pair<torch::Tensor, torch::optional<torch::Tensor>>
CudaImpl(torch::Tensor activations, torch::Tensor targets,
         torch::Tensor activation_lengths, torch::Tensor target_lengths,
         int32_t blank, float fastemit_lambda) {
#ifdef RNNT_WITH_CUDA
  // Check devices
  TORCH_CHECK(activations.device().is_cuda(), "activations.device() ",
              activations.device(), " is not CUDA");

  TORCH_CHECK(activation_lengths.device().is_cuda(),
              "activation_lengths.device() ", activation_lengths.device(),
              " is not CUDA");

  TORCH_CHECK(target_lengths.device().is_cuda(), "target_lengths.device() ",
              target_lengths.device(), " is not CUDA");

  TORCH_CHECK(targets.device().is_cuda(), "targets.device() ", targets.device(),
              " is not CUDA");

  // activations.shape is (N, T, U, C)
  int32_t N = activations.size(0);
  int32_t T = activations.size(1);
  int32_t U = activations.size(2);
  int32_t C = activations.size(3);

  rnntOptions options{};
  options.loc = RNNT_GPU;
  options.num_threads = 1;
  options.stream = c10::cuda::getCurrentCUDAStream();
  options.blank_label = blank;
  options.maxT = T;
  options.maxU = U;
  options.batch_first = true;
  options.fastemit_lambda = fastemit_lambda;

  torch::optional<torch::Tensor> grads;
  if (activations.requires_grad()) {
    grads = torch::empty_like(activations);
  }

  // costs is always on CPU
  torch::Tensor costs = torch::empty({N}, activations.dtype());

  size_t gpu_size_bytes = 0;

  if (activations.scalar_type() == torch::kFloat) {
    get_workspace_size(T, U, N,
                       /*gpu*/ true, &gpu_size_bytes, sizeof(float));
    torch::Tensor workspace =
        torch::empty({static_cast<int32_t>(gpu_size_bytes)},
                     torch::device(activations.device()).dtype(torch::kByte));

    compute_rnnt_loss(
        activations.data_ptr<float>(),
        grads ? grads->data_ptr<float>() : nullptr, targets.data_ptr<int>(),
        target_lengths.data_ptr<int>(), activation_lengths.data_ptr<int>(), C,
        N, costs.data_ptr<float>(), workspace.data_ptr<uint8_t>(), options);
  } else {
    get_workspace_size(T, U, N,
                       /*gpu*/ true, &gpu_size_bytes, sizeof(double));
    torch::Tensor workspace =
        torch::empty({static_cast<int32_t>(gpu_size_bytes)},
                     torch::device(activations.device()).dtype(torch::kByte));

    compute_rnnt_loss_fp64(
        activations.data_ptr<double>(),
        grads ? grads->data_ptr<double>() : nullptr, targets.data_ptr<int>(),
        target_lengths.data_ptr<int>(), activation_lengths.data_ptr<int>(), C,
        N, costs.data_ptr<double>(), workspace.data_ptr<uint8_t>(), options);
  }

  costs = costs.to(activations.device());
  return {costs, grads};
#else
  TORCH_CHECK(false, "warp-rnnt was not compiled with CUDA!\n",
              "Please install a CUDA version.\n",
              "Refer to https://github.com/csukuangfj/warp-transducer for "
              "installation.");
  return {};
#endif
}

/**
 @params activations If on CPU, it should be the output of a log-softmax layer;
                if on CUDA, it should be the output of a linear layer.
                Its shape is (N, T, U, C), where N is the batch size, T means
                number of frames, U is `max_target_length` + 1, C
                is the number of output classes.
                It should be contiguous in memory. Supported dtypes are
                torch.float32 and torch.float64.

  @params targets A 2-D tensor of shape (N, U) containing targets with zero
                  padded. It should be contiguous in memory. Only torch.int32
                  is supported.
  @params activation_lengths A 1-D tensor of shape (N,). It should be contiguous
                        in memory. Only torch.int32 is supported. It contains
                        lengths of each sequence from the encoder.
  @params target_lengths A 1-D tensor of shape (N,). It should be contiguous
                         in memory. Only torch.int32 is supported. It contains
                         lengths of targets of each sequence from the prediction
                         network.
  @params blank  The ID of the blank symbol
  @params fastemit_lambda Regularization parameter for FastEmit

  @returns Return a pair containing two tensors:

    - loss, its shape is (N,) with the same dtype of activations.
    - grad, if activations.requires_grad() is true, this tensor
            contains the gradients for activations.
 */
static std::pair<torch::Tensor, torch::optional<torch::Tensor>>
ComputeRnntLoss(torch::Tensor activations, torch::Tensor targets,
                torch::Tensor activation_lengths, torch::Tensor target_lengths,
                int32_t blank, float fastemit_lambda) {
  // First, for CPU only
  TORCH_CHECK(activations.dim() == 4, "activations.dim() ", activations.dim(),
              " is not 4");
  TORCH_CHECK(activations.is_contiguous(), "activations is not contiguous");
  TORCH_CHECK((activations.scalar_type() == torch::kFloat) ||
                  (activations.scalar_type() == torch::kDouble),
              "activations.scalar_type() ", activations.scalar_type(),
              " is not Float or Double");

  TORCH_CHECK(targets.dim() == 2, "targets.dim() ", targets.dim(), " is not 2");
  TORCH_CHECK(targets.is_contiguous(), "targets is not contiguous");
  TORCH_CHECK(targets.scalar_type() == torch::kInt, "targets.scalar_type() ",
              targets.scalar_type(), " is not kInt");

  TORCH_CHECK(activation_lengths.dim() == 1, "activation_lengths.dim() ",
              activation_lengths.dim(), " is not 1");
  TORCH_CHECK(activation_lengths.is_contiguous(),
              "activation_lengths is not contiguous");
  TORCH_CHECK(activation_lengths.scalar_type() == torch::kInt,
              "activation_lengths.scalar_type() ",
              activation_lengths.scalar_type(), " is not kInt");

  TORCH_CHECK(target_lengths.dim() == 1, "target_lengths.dim() ",
              target_lengths.dim(), " is not 1");
  TORCH_CHECK(target_lengths.is_contiguous(),
              "target_lengths is not contiguous");
  TORCH_CHECK(target_lengths.scalar_type() == torch::kInt,
              "target_lengths.scalar_type() ", target_lengths.scalar_type(),
              " is not kInt");

  // Check that shapes match
  TORCH_CHECK(activations.size(0) == targets.size(0),
              "activations.size(0) != targets.size(0), i.e., ",
              activations.size(0), " != ", targets.size(0));

  TORCH_CHECK(activations.size(0) == activation_lengths.size(0),
              "activations.size(0) != activation_lengths.size(0), i.e., ",
              activations.size(0), " != ", activation_lengths.size(0));

  TORCH_CHECK(activations.size(0) == target_lengths.size(0),
              "activations.size(0) != target_lengths.size(0), i.e., ",
              activations.size(0), " != ", target_lengths.size(0));

  // activations.shape is (N, T, U, C)
  int32_t T = activations.size(1);
  int32_t U = activations.size(2);
  int32_t C = activations.size(3);

  TORCH_CHECK(blank >= 0 && blank < C, "blank is ", blank,
              ", which is not in the range [0, ", C, ")");

  TORCH_CHECK(T == activation_lengths.max().item().toInt(),
              "activations.size(1) vs activation_lengths.max(): ", T, " vs ",
              activation_lengths.max().item().toInt());

  TORCH_CHECK(U == target_lengths.max().item().toInt() + 1,
              "activations.size(2) vs target_lengths.max() + 1: ", U, " vs ",
              target_lengths.max().item().toInt() + 1);

  TORCH_CHECK(targets.size(1) == target_lengths.max().item().toInt(),
              "targets.size(1) vs target_lengths.max() : ", targets.size(1),
              " vs ", target_lengths.max().item().toInt());

  if (activations.device().is_cpu()) {
    return CpuImpl(activations, targets, activation_lengths, target_lengths,
                   blank, fastemit_lambda);
  } else {
    return CudaImpl(activations, targets, activation_lengths, target_lengths,
                    blank, fastemit_lambda);
  }
}

PYBIND11_MODULE(_warp_rnnt_py, m) {
  m.doc() = R"(PyTorch Python wrapper of warp-rnnt.
See
 - https://github.com/csukuangfj/warp-transducer
 - https://github.com/b-flo/warp-transducer
 - https://github.com/HawkAaron/warp-transducer

If you have any questions, please file an issue at

  https://github.com/csukuangfj/warp-transducer/issues/new

)";

  m.attr("version") = get_warprnnt_version();
  m.def("rnnt_loss", &ComputeRnntLoss, py::arg("activations"),
        py::arg("targets"), py::arg("activation_lengths"),
        py::arg("target_lengths"), py::arg("blank"),
        py::arg("fastemit_lambda"));
}
