import importlib.util
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy import ndimage


class SegmentationEngine:
    MODES = {"strict", "adaptive", "lenient"}

    def __init__(self):
        self.model_dir = Path(
            os.path.expanduser(os.getenv("U2NET_HOME", "~/.u2net"))
        )
        self.available = (
            importlib.util.find_spec("onnxruntime") is not None
            and (self.model_dir / "isnet-general-use.onnx").exists()
            and (self.model_dir / "u2net.onnx").exists()
        )
        self._ort = None
        self._sessions = {}
        self._lock = threading.RLock()

    def _ensure_runtime(self):
        if not self.available:
            raise RuntimeError("ONNX Runtime or local segmentation weights are unavailable.")
        if self._ort is None:
            with self._lock:
                if self._ort is None:
                    os.environ.setdefault("OMP_NUM_THREADS", "4")
                    import onnxruntime as ort

                    self._ort = ort

    def _get_session(self, model_name):
        self._ensure_runtime()
        if model_name not in self._sessions:
            with self._lock:
                if model_name not in self._sessions:
                    model_path = self.model_dir / f"{model_name}.onnx"
                    if not model_path.exists():
                        raise FileNotFoundError(f"Missing segmentation model: {model_path}")
                    options = self._ort.SessionOptions()
                    options.inter_op_num_threads = 1
                    options.intra_op_num_threads = 4
                    options.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
                    options.graph_optimization_level = (
                        self._ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                    )
                    self._sessions[model_name] = self._ort.InferenceSession(
                        str(model_path),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
        return self._sessions[model_name]

    def _predict_alpha(self, image, model_name):
        session = self._get_session(model_name)
        if model_name == "isnet-general-use":
            size = (1024, 1024)
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif model_name == "u2net":
            size = (320, 320)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported segmentation model: {model_name}")

        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32)
        array /= max(float(array.max()), 1e-6)
        array = (array - mean) / std
        model_input = np.expand_dims(array.transpose(2, 0, 1), 0).astype(np.float32)
        input_name = session.get_inputs()[0].name
        prediction = session.run(None, {input_name: model_input})[0][:, 0, :, :]
        minimum = float(prediction.min())
        maximum = float(prediction.max())
        prediction = (prediction - minimum) / max(maximum - minimum, 1e-6)
        mask = Image.fromarray(
            (np.squeeze(prediction).clip(0, 1) * 255).astype(np.uint8), mode="L"
        )
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        return np.asarray(mask, dtype=np.float32) / 255.0

    @staticmethod
    def _enhance_dark_image(image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gamma_corrected = np.power(array, 0.58)
        enhanced = Image.fromarray(
            np.clip(gamma_corrected * 255.0, 0, 255).astype(np.uint8), mode="RGB"
        )
        enhanced = ImageOps.autocontrast(enhanced, cutoff=1)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
        return ImageEnhance.Color(enhanced).enhance(1.05)

    @staticmethod
    def thresholds(sensitivity):
        sensitivity = min(max(int(sensitivity), 0), 100)
        ratio = 0.015 + (0.00045 * sensitivity)
        center = 0.02 + (0.0008 * sensitivity)
        median_alpha = 0.12 + (0.0011 * sensitivity)
        high_confidence = 0.04 + (0.00085 * sensitivity)
        return ratio, center, median_alpha, high_confidence

    @classmethod
    def analyze_alpha(cls, alpha, sensitivity=70):
        if isinstance(alpha, Image.Image):
            alpha = np.asarray(alpha.convert("L"), dtype=np.float32) / 255.0
        alpha = np.asarray(alpha, dtype=np.float32)
        height, width = alpha.shape
        center = alpha[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
        border_width = max(1, int(min(height, width) * 0.05))
        border_values = np.concatenate([
            alpha[:border_width, :].reshape(-1),
            alpha[-border_width:, :].reshape(-1),
            alpha[border_width:-border_width, :border_width].reshape(-1),
            alpha[border_width:-border_width, -border_width:].reshape(-1),
        ])
        foreground_ratio = float((alpha > 0.05).mean())
        center_ratio = float((center > 0.05).mean()) if center.size else 0.0
        edge_foreground_ratio = float((border_values > 0.10).mean())
        visible_values = alpha[alpha > 0.02]
        visible_ratio = float((alpha > 0.02).mean())
        if visible_values.size:
            median_visible_alpha = float(np.median(visible_values))
            high_confidence_fraction = float((visible_values > 0.80).mean())
            translucent_fraction = float((visible_values < 0.80).mean())
        else:
            median_visible_alpha = 0.0
            high_confidence_fraction = 0.0
            translucent_fraction = 1.0

        min_ratio, min_center, min_median_alpha, min_high_confidence = (
            cls.thresholds(sensitivity)
        )
        opacity_suspicious = visible_ratio > 0.03 and (
            median_visible_alpha < min_median_alpha
            or high_confidence_fraction < min_high_confidence
        )
        area_score = min(foreground_ratio / max(min_ratio, 1e-6), 1.0)
        center_score = min(center_ratio / max(min_center, 1e-6), 1.0)
        opacity_score = min(
            median_visible_alpha / max(min_median_alpha, 1e-6), 1.0
        )
        core_score = min(
            high_confidence_fraction / max(min_high_confidence, 1e-6), 1.0
        )
        contamination_score = max(
            0.0, 1.0 - max(0.0, edge_foreground_ratio - 0.25) / 0.75
        )
        base_quality = 0.25 * (
            area_score + center_score + opacity_score + core_score
        )
        return {
            "foreground_ratio": foreground_ratio,
            "center_foreground_ratio": center_ratio,
            "mean_alpha": float(alpha.mean()),
            "visible_foreground_ratio": visible_ratio,
            "median_visible_alpha": median_visible_alpha,
            "high_confidence_fraction": high_confidence_fraction,
            "translucent_fraction": translucent_fraction,
            "edge_foreground_ratio": edge_foreground_ratio,
            "background_contamination_score": contamination_score,
            "quality_score": float(base_quality * (0.70 + 0.30 * contamination_score)),
            "suspicious": bool(
                foreground_ratio < min_ratio
                or center_ratio < min_center
                or opacity_suspicious
            ),
            "minimum_foreground_ratio": float(min_ratio),
            "minimum_center_ratio": float(min_center),
            "minimum_median_alpha": float(min_median_alpha),
            "minimum_high_confidence_fraction": float(min_high_confidence),
        }

    @staticmethod
    def _keep_relevant_components(alpha, strict_alpha):
        binary = alpha > 0.03
        labels, component_count = ndimage.label(binary)
        if component_count == 0:
            return alpha

        height, width = alpha.shape
        center = np.zeros_like(binary)
        center[height // 5: 4 * height // 5, width // 5: 4 * width // 5] = True
        strict_binary = strict_alpha > 0.05
        minimum_area = max(16, int(alpha.size * 0.0005))
        keep = np.zeros_like(binary)

        for component_id in range(1, component_count + 1):
            component = labels == component_id
            area = int(component.sum())
            if area < minimum_area:
                continue
            if (component & center).any() or (component & strict_binary).any():
                keep |= component

        return np.where(keep, alpha, 0.0)

    def segment(self, image, mode="adaptive", sensitivity=70):
        if mode not in self.MODES:
            raise ValueError(f"Unsupported segmentation mode: {mode}")
        sensitivity = min(max(int(sensitivity), 0), 100)
        source = image.convert("RGB")

        strict_alpha = self._predict_alpha(source, "isnet-general-use")
        strict_metrics = self.analyze_alpha(strict_alpha, sensitivity)
        should_recover = mode == "lenient" or (
            mode == "adaptive" and strict_metrics["suspicious"]
        )

        final_alpha = strict_alpha
        passes = ["isnet-general-use"]
        if should_recover:
            enhanced = self._enhance_dark_image(source)
            enhanced_isnet = self._predict_alpha(enhanced, "isnet-general-use")
            enhanced_u2net = self._predict_alpha(enhanced, "u2net")
            passes.extend(["isnet-enhanced", "u2net-enhanced"])

            isnet_alpha = np.maximum(strict_alpha, enhanced_isnet)
            final_alpha = np.maximum(isnet_alpha, enhanced_u2net)
            weak_threshold = 0.04 - (0.00025 * sensitivity)
            strong_threshold = 0.20 - (0.001 * sensitivity)
            cross_model_agreement = (
                (isnet_alpha > weak_threshold)
                & (enhanced_u2net > weak_threshold)
            )
            if mode == "adaptive":
                reliable = (final_alpha > strong_threshold) | cross_model_agreement
                final_alpha = np.where(reliable, final_alpha, 0.0)
            gamma = 0.90 - (0.0045 * sensitivity)
            final_alpha = np.power(np.clip(final_alpha, 0.0, 1.0), gamma)

            dilation_size = 1 if sensitivity < 45 else (3 if sensitivity < 85 else 5)
            if dilation_size > 1:
                final_alpha = ndimage.maximum_filter(final_alpha, size=dilation_size)
            final_alpha = self._keep_relevant_components(final_alpha, strict_alpha)

            recovered_metrics = self.analyze_alpha(final_alpha, sensitivity)
            recovery_improved = (
                recovered_metrics["quality_score"]
                > strict_metrics["quality_score"] + 0.02
            )
            if (
                recovered_metrics["foreground_ratio"] > 0.92
                or (mode == "adaptive" and not recovery_improved)
            ):
                final_alpha = strict_alpha
                should_recover = False

        final_metrics = self.analyze_alpha(final_alpha, sensitivity)
        final_metrics.update({
            "mode": mode,
            "sensitivity": sensitivity,
            "recovery_used": bool(should_recover),
            "suspicious_before": bool(strict_metrics["suspicious"]),
            "model_passes": passes,
        })

        rgba = source.convert("RGBA")
        alpha_image = Image.fromarray(
            np.clip(final_alpha * 255.0, 0, 255).astype(np.uint8), mode="L"
        )
        rgba.putalpha(alpha_image)
        return rgba, final_metrics


segmentation_engine = SegmentationEngine()
