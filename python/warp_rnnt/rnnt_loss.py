# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)

import torch

import _warp_rnnt_py


class RnntLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        activations: torch.Tensor,
        targets: torch.Tensor,
        activation_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Args:
          activations:
            A 4-D tensor of shape (N, T, U_plus_1, C) with dtype torch.float32
            or torch.float64. If it is on CPU, it should be the output of
            a log-softmax layer; if it is on CUDA, it should be the output of
            a linear layer, i.e., logits without normalization. `N` is the batch
            size; `T` is the number of output frames from the encoder; `U` is
            the max target length; `C` is the vocabulary size.
            It must be contiguous in memory.
          targets:
            A 2-D tensor of shape (N, U) with dtype torch.int32. Should always
            be on CPU and contiguous in memory.
          activation_lengths:
            A 1-D tensor of shape (N,) with dtype torch.int32. Should always
            be on CPU and contiguous in memory.
          target_lengths:
            A 1-D tensor of shape (N,) with dtype torch.int32. Should always
            be on CPU and contiguous in memory.
          blank:
            The ID for the blank symbol.
          reduction:
            Supported values are:

                - "none". Return a tensor of shape (N,) containing the losses
                          for each utterance in the batch.
                - "mean". Return a tensor of shape (1,) containing the average
                          loss over all utterances in the batch.
                - "sum". Return a tensor of shape (1,) containing the sum of
                          the loss over all utterances in the batch.
        """
        assert reduction in ("none", "mean", "sum")

        loss, grad = _warp_rnnt_py.rnnt_loss(
            activations=activations,
            targets=targets,
            activation_lengths=activation_lengths,
            target_lengths=target_lengths,
            blank=blank,
            fastemit_lambda=0,
        )

        if reduction == "mean":
            loss = loss.mean()
            grad /= activations.size(0)
        elif reduction == "sum":
            loss = loss.sum()

        ctx.grad = grad

        return loss

    @staticmethod
    def backward(ctx, loss_grad):
        loss_grad = loss_grad.reshape(-1, 1, 1, 1)
        return (
            ctx.grad.mul_(loss_grad),  # activations
            None,  # targets,
            None,  # activation_lengths,
            None,  # target_lengths
            None,  # blank
            None,  # reduction
        )


class RnntLoss(torch.nn.Module):
    def __init__(self, blank: int, reduction: str = "mean"):
        """
        Args:
          blank:
            The ID for the blank symbol.
          reduction:
            Supported values are:

                - "none". Return a tensor of shape (N,) containing the losses
                          for each utterance in the batch.
                - "mean". Return a tensor of shape (1,) containing the average
                          loss over all utterances in the batch.
                - "sum". Return a tensor of shape (1,) containing the sum of
                          the loss over all utterances in the batch.
        """
        self.blank = blank
        self.reduction = reduction

    def forward(
        self,
        activations: torch.Tensor,
        targets: torch.Tensor,
        activation_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          activations:
            A 4-D tensor of shape (N, T, U_plus_1, C) with dtype torch.float32
            or torch.float64. If it is on CPU, it should be the output of
            a log-softmax layer; if it is on CUDA, it should be the output of
            a linear layer, i.e., logits without normalization. `N` is the batch
            size; `T` is the number of output frames from the encoder; `U` is
            the max target length; `C` is the vocabulary size.
            It must be contiguous in memory.
          targets:
            A 2-D tensor of shape (N, U) with dtype torch.int32. Should always
            be on CPU and contiguous in memory.
          activation_lengths:
            A 1-D tensor of shape (N,) with dtype torch.int32. Should always
            be on CPU and contiguous in memory.
          target_lengths:
            A 1-D tensor of shape (N,) with dtype torch.int32. Should always
            be on CPU and contiguous in memory.
        """
        if activations.device.type == "cpu":
            activations = activations.log_softmax(-1)
        return RnntLossFunction.apply(
            activations,
            targets,
            activation_lengths,
            target_lengths,
            self.blank,
            self.reduction,
        )


def rnnt_loss(
    activations: torch.Tensor,
    targets: torch.Tensor,
    activation_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Args:
      activations:
        A 4-D tensor of shape (N, T, U_plus_1, C) with dtype torch.float32
        or torch.float64. If it is on CPU, it should be the output of
        a log-softmax layer; if it is on CUDA, it should be the output of
        a linear layer, i.e., logits without normalization. `N` is the batch
        size; `T` is the number of output frames from the encoder; `U` is
        the max target length; `C` is the vocabulary size.
        It must be contiguous in memory.
      targets:
        A 2-D tensor of shape (N, U) with dtype torch.int32. Should always
        be contiguous in memory.
      activation_lengths:
        A 1-D tensor of shape (N,) with dtype torch.int32. Should always
        be contiguous in memory.
      target_lengths:
        A 1-D tensor of shape (N,) with dtype torch.int32. Should always
        be contiguous in memory.
      blank:
        The ID for the blank symbol.
      reduction:
        Supported values are:

            - "none". Return a tensor of shape (N,) containing the losses
                      for each utterance in the batch.
            - "mean". Return a tensor of shape (1,) containing the average
                      loss over all utterances in the batch.
            - "sum". Return a tensor of shape (1,) containing the sum of
                      the loss over all utterances in the batch.
    """
    if activations.device.type == "cpu":
        activations = activations.log_softmax(-1)
    return RnntLossFunction.apply(
        activations,
        targets,
        activation_lengths,
        target_lengths,
        blank,
        reduction,
    )
