"""Customer segmentation package."""

from .data_loader import load_segmentation_data, load_purchase_data
from .segmentation import SegmentationPipeline

__all__ = ["load_segmentation_data", "load_purchase_data", "SegmentationPipeline"]
