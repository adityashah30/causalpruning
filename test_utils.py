from dataclasses import dataclass
import torch
import torchmetrics


@dataclass
class EvalMetrics:
    accuracy: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1_score: torch.Tensor


class MetricsComputer:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.f1_score = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="none"
        )

    def to(self, device: torch.device) -> "MetricsComputer":
        self.accuracy.to(device)
        self.precision.to(device)
        self.recall.to(device)
        self.f1_score.to(device)
        return self

    def reset(self) -> "MetricsComputer":
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()
        return self

    def add(self, logits: torch.Tensor, labels: torch.Tensor):
        self.accuracy(logits, labels)
        self.precision(logits, labels)
        self.recall(logits, labels)
        self.f1_score(logits, labels)

    def compute(self) -> EvalMetrics:
        return EvalMetrics(
            accuracy=self.accuracy.compute(),
            precision=self.precision.compute(),
            recall=self.recall.compute(),
            f1_score=self.f1_score.compute(),
        )
