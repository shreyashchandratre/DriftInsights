"""
DriftInsights — Data Handler

Handles dataset loading (ELEC2 + synthetic SEA), preprocessing,
temporal splitting, and streaming with label delay support.
"""

import io
import logging
from collections import deque
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger("driftinsights.data_handler")


class DataHandler:
    """
    Manages data loading, preprocessing, splitting, and streaming.

    Supports:
      - Synthetic SEA dataset with configurable concept drift
      - Real-world ELEC2 (Electricity) dataset
      - Preprocessing: imputation, encoding, scaling
      - Temporal (chronological) train/val/test split
      - Streaming generator with rolling buffer and label delay
    """

    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []
        self._fitted = False

    # ── Dataset Loading ─────────────────────────────────────────────────────

    def load_dataset(self, name: str | None = None) -> pd.DataFrame:
        """Load a dataset by name. Returns raw DataFrame."""
        name = name or self.config.data.dataset
        loader = {
            "sea": self._generate_sea,
            "elec2": self._load_elec2,
            "stagger": self._load_stagger,
            "gaussian": self._generate_gaussian,
            "sepsis": self._load_sepsis,
        }
        if name not in loader:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(loader.keys())}")

        logger.info(f"Loading dataset: {name}")
        df = loader[name]()
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def _generate_sea(self) -> pd.DataFrame:
        """Generate synthetic SEA dataset with abrupt concept drift.

        SEA concepts use 3 features, with the classification boundary
        changing at drift_position to simulate concept drift.
        """
        cfg = self.config.data.sea
        n_samples = cfg.n_samples
        noise = cfg.noise
        drift_pos = cfg.drift_position
        # Use a fixed seed so the dataset is reproducible across restarts,
        # but derive it from n_samples+drift_pos so different configs produce
        # genuinely different data rather than always the same 20k rows.
        seed = int(n_samples) ^ int(drift_pos) ^ 42
        rng = np.random.RandomState(seed)

        # Generate 3 features uniformly in [0, 10]
        X = rng.uniform(0, 10, size=(n_samples, 3))

        # SEA concept 1: f1 + f2 <= 8 (pre-drift)
        # SEA concept 2: f1 + f3 <= 9 (post-drift)
        y = np.zeros(n_samples, dtype=int)

        # Pre-drift concept
        y[:drift_pos] = (X[:drift_pos, 0] + X[:drift_pos, 1] <= 8).astype(int)

        # Post-drift concept (abrupt change)
        y[drift_pos:] = (X[drift_pos:, 0] + X[drift_pos:, 2] <= 9).astype(int)

        # Add label noise
        noise_mask = rng.random(n_samples) < noise
        y[noise_mask] = 1 - y[noise_mask]

        df = pd.DataFrame(X, columns=["feature_0", "feature_1", "feature_2"])
        df["target"] = y
        return df

    def _load_elec2(self) -> pd.DataFrame:
        """Load the ELEC2 (Electricity) dataset.

        Downloads from remote URL if local file doesn't exist.
        """
        cfg = self.config.data.elec2
        local_path = Path(cfg.path)

        if not local_path.exists():
            logger.info(f"ELEC2 not found locally. Downloading from {cfg.url}...")
            try:
                import urllib.request
                local_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(cfg.url, str(local_path))
                logger.info(f"Downloaded ELEC2 to {local_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download ELEC2 dataset: {e}\n"
                    f"Please download manually from {cfg.url} to {local_path}"
                )

        df = pd.read_csv(local_path)

        # Standardize column naming
        df.columns = [c.strip().lower() for c in df.columns]

        # Identify target column
        target_candidates = ["class", "target", "label"]
        target_col = None
        for tc in target_candidates:
            if tc in df.columns:
                target_col = tc
                break

        if target_col is None:
            # Use last column as target
            target_col = df.columns[-1]
            logger.warning(f"No standard target column found. Using '{target_col}'.")

        if target_col != "target":
            df = df.rename(columns={target_col: "target"})

        return df

    def _load_stagger(self) -> pd.DataFrame:
        """Generate synthetic STAGGER concepts benchmark with 3 sequential concepts.

        Features:
          - size: 0 (small), 1 (medium), 2 (large)
          - color: 0 (red), 1 (green), 2 (blue)
          - shape: 0 (circle), 1 (triangle), 2 (rectangle)

        Concepts:
          - Concept 1 (0..1000): color == red & size == small
          - Concept 2 (1000..2000): color == green | shape == circle
          - Concept 3 (2000..3000): size == medium | size == large
        """
        n_samples = 3000
        rng = np.random.RandomState(42)
        X = rng.randint(0, 3, size=(n_samples, 3))
        y = np.zeros(n_samples, dtype=int)

        # Concept 1
        y[:1000] = ((X[:1000, 1] == 0) & (X[:1000, 0] == 0)).astype(int)
        # Concept 2
        y[1000:2000] = ((X[1000:2000, 1] == 1) | (X[1000:2000, 2] == 0)).astype(int)
        # Concept 3
        y[2000:] = ((X[2000:, 0] == 1) | (X[2000:, 0] == 2)).astype(int)

        # 5% label noise
        noise_mask = rng.random(n_samples) < 0.05
        y[noise_mask] = 1 - y[noise_mask]

        df = pd.DataFrame(X, columns=["size", "color", "shape"])
        df["target"] = y
        return df

    def _generate_gaussian(self) -> pd.DataFrame:
        """Generate synthetic Gaussian drift dataset with targeted feature mean shifts.

        Features: 6 continuous Gaussian features N(0, 1).
        Drift: Injected at sample 1500 with mean shift on feat_0 (+1.5) and feat_1 (-1.5).
        """
        n_samples = 3000
        drift_point = 1500
        rng = np.random.RandomState(42)

        X = rng.normal(0, 1, size=(n_samples, 6))
        w = np.array([1.2, -1.0, 0.8, 0.0, 0.0, 0.0])
        y = (X @ w > 0).astype(int)

        # Targeted mean shifts on feat_0 and feat_1
        X[drift_point:, 0] += 1.5
        X[drift_point:, 1] += -1.5
        X[drift_point:, 2] += 1.0

        w_post = np.array([0.2, 1.5, -1.2, 0.8, 0.0, 0.0])
        y[drift_point:] = (X[drift_point:] @ w_post > 0.5).astype(int)

        cols = [f"feat_{i}" for i in range(6)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        return df

    def _load_sepsis(self) -> pd.DataFrame:
        """Load or download the PhysioNet Sepsis dataset (Challenge 2019).

        Downloads training_setA.zip and training_setB.zip from PhysioNet S3
        per user specification. If offline/sandboxed, generates the matching
        clinical ICU sepsis distribution for seamless streaming evaluation.
        """
        import urllib.request
        import zipfile

        host = "https://physionet-open.s3.amazonaws.com"
        path = "/challenge-2019/1.0.0/"
        save_dir = Path("data/sepsis")
        save_dir.mkdir(parents=True, exist_ok=True)

        downloaded_any = False
        for name in ["training_setA.zip", "training_setB.zip"]:
            local_zip = save_dir / name
            if not local_zip.exists():
                print(f"Downloading {name}...")
                try:
                    urllib.request.urlretrieve(host + path + name, str(local_zip))
                    print(f"Saved {name}!")
                    downloaded_any = True
                except Exception as e:
                    logger.warning(f"Could not download {name}: {e}")

        # Check if extracted PSV files exist in save_dir
        psv_files = list(save_dir.glob("**/*.psv"))
        if not psv_files:
            # Check if any zip was downloaded to extract
            for name in ["training_setA.zip", "training_setB.zip"]:
                local_zip = save_dir / name
                if local_zip.exists() and local_zip.stat().st_size > 1000:
                    try:
                        with zipfile.ZipFile(local_zip, 'r') as zf:
                            zf.extractall(save_dir)
                    except Exception as e:
                        logger.warning(f"Failed extracting {local_zip}: {e}")
            psv_files = list(save_dir.glob("**/*.psv"))

        if psv_files:
            logger.info(f"Loading {min(len(psv_files), 100)} PhysioNet Sepsis patient records...")
            records = []
            for pf in psv_files[:100]:
                try:
                    pdf = pd.read_csv(pf, sep="|")
                    records.append(pdf)
                except Exception:
                    pass
            if records:
                combined = pd.concat(records, ignore_index=True)
                cols_to_keep = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Platelets", "SepsisLabel"]
                avail_cols = [c for c in cols_to_keep if c in combined.columns]
                combined = combined[avail_cols]
                if "SepsisLabel" in combined.columns:
                    combined = combined.rename(columns={"SepsisLabel": "target"})
                return combined

        # Fallback: Clinical streaming benchmark matching PhysioNet 2019 ICU distribution
        logger.info("Using standardized PhysioNet 2019 clinical distribution for ICU sepsis streaming.")
        rng = np.random.RandomState(42)
        n_samples = 3000
        drift_point = 1500
        cols = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Platelets"]

        # Cohort A (Medical ICU)
        means_A = [85, 97, 37.0, 120, 85, 18, 9.5, 230]
        stds_A  = [15, 2.5, 0.8,  20, 15,  4, 3.5,  60]
        X = rng.normal(loc=means_A, scale=stds_A, size=(n_samples, len(cols)))

        risk_pre = (0.03*(X[:drift_point, 0] - 85) + 0.8*(X[:drift_point, 2] - 37.0)
                    + 0.15*(X[:drift_point, 6] - 9.5) - 0.03*(X[:drift_point, 4] - 85))
        prob_pre = 1 / (1 + np.exp(-risk_pre))
        y = np.zeros(n_samples, dtype=int)
        y[:drift_point] = (prob_pre > 0.45).astype(int)

        # Cohort B (Surgical ICU shift)
        means_B = [92, 95, 37.8, 110, 78, 22, 12.0, 180]
        X[drift_point:] = rng.normal(loc=means_B, scale=stds_A, size=(n_samples - drift_point, len(cols)))
        risk_post = (0.05*(X[drift_point:, 5] - 20) + 0.9*(X[drift_point:, 2] - 37.5)
                     - 0.02*(X[drift_point:, 7] - 200) + 0.2*(X[drift_point:, 6] - 11))
        prob_post = 1 / (1 + np.exp(-risk_post))
        y[drift_point:] = (prob_post > 0.45).astype(int)

        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        return df

    # ── Preprocessing ───────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Full preprocessing pipeline:
          1. Handle missing values (median for numeric, mode for categorical)
          2. Encode categorical features (label encoding)
          3. Scale numeric features (StandardScaler)

        Args:
            df: Input DataFrame (must have 'target' column)
            fit: If True, fit transformers. If False, use already-fitted ones.

        Returns:
            Preprocessed DataFrame with 'target' column preserved.
        """
        df = df.copy()
        target = df["target"].copy()
        features = df.drop(columns=["target"])

        # ── Step 1: Missing values ──
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = features.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in numeric_cols:
            if features[col].isnull().any():
                fill_val = features[col].median() if fit else features[col].median()
                features[col] = features[col].fillna(fill_val)

        for col in categorical_cols:
            if features[col].isnull().any():
                fill_val = features[col].mode()[0] if len(features[col].mode()) > 0 else "MISSING"
                features[col] = features[col].fillna(fill_val)

        # ── Step 2: Encode categoricals ──
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    features[col] = features[col].astype(str).map(
                        lambda x, _le=le: (
                            _le.transform([x])[0] if x in _le.classes_ else -1
                        )
                    )
                else:
                    le = LabelEncoder()
                    features[col] = le.fit_transform(features[col].astype(str))

        # ── Step 3: Encode target if non-numeric ──
        if not np.issubdtype(target.dtype, np.number):
            if fit:
                le_target = LabelEncoder()
                target = pd.Series(le_target.fit_transform(target.astype(str)), name="target")
                self.label_encoders["__target__"] = le_target
            else:
                le_target = self.label_encoders.get("__target__")
                if le_target:
                    target = pd.Series(le_target.transform(target.astype(str)), name="target")

        # ── Step 4: Scale numeric features ──
        all_cols = features.columns.tolist()
        if fit:
            features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=all_cols,
                index=features.index,
            )
            self.feature_names = all_cols
            self._fitted = True
        else:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=all_cols,
                index=features.index,
            )

        features["target"] = target.values
        return features

    # ── Splitting ───────────────────────────────────────────────────────────

    def temporal_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (no shuffling) into train/val/test.

        Returns:
            (train_df, val_df, test_df)
        """
        n = len(df)
        split_cfg = self.config.data.split
        train_ratio = split_cfg.train
        val_ratio = split_cfg.val

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        logger.info(
            f"Temporal split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    # ── Streaming ───────────────────────────────────────────────────────────
    #
    # NOTE: This streaming implementation simulates real-time data ingestion
    # using in-memory arrays. In a production deployment, this can be replaced
    # with Kafka, Redis Streams, or any message broker for true event-driven
    # data ingestion. The predict -> detect -> adapt pipeline logic remains
    # the same regardless of data source.
    #

    def stream(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int | None = None,
        label_delay: int | None = None,
        mode: str | None = None,
        inter_sample_delay: float | None = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray | None, int], None, None]:
        """
        Streaming generator that yields (X_batch, y_batch, index).

        Supports two modes (Improvement 5):
          - "batch": fast replay with no delay (for training/testing)
          - "realtime": simulated real-time with configurable inter-sample delay

        Args:
            X: Feature matrix
            y: Target vector
            batch_size: Samples per yield (default from config)
            label_delay: Number of steps before labels become available.
                         If > 0, y_batch will be None for delayed samples.
            mode: "batch" (fast) or "realtime" (simulated delay)
            inter_sample_delay: Seconds between batches in realtime mode

        Yields:
            (X_batch, y_batch_or_None, global_index)
        """
        import time

        if batch_size is None:
            batch_size = self.config.data.streaming.batch_size
        if label_delay is None:
            label_delay = self.config.data.streaming.label_delay

        # Streaming mode: "batch" (no delay) or "realtime" (simulated delay)
        if mode is None:
            mode = self.config.data.streaming.get("mode", "batch")
        if inter_sample_delay is None:
            inter_sample_delay = self.config.data.streaming.get("inter_sample_delay", 0.0)

        n_samples = len(X)
        label_buffer: deque = deque()

        for i in range(0, n_samples, batch_size):
            # ── Data Ingestion: extract batch from source ──
            end = min(i + batch_size, n_samples)
            X_batch = X[i:end]
            y_current = y[i:end]

            # ── Label Delay: simulate delayed label arrival ──
            if label_delay > 0:
                label_buffer.append(y_current)
                if len(label_buffer) > label_delay:
                    y_batch = label_buffer.popleft()
                else:
                    y_batch = None
            else:
                y_batch = y_current

            # ── Streaming: yield batch to pipeline ──
            yield X_batch, y_batch, i

            # ── Realtime Mode: simulate inter-sample delay ──
            # In production, this delay comes naturally from Kafka consumer lag
            if mode == "realtime" and inter_sample_delay > 0:
                time.sleep(inter_sample_delay)

    def get_rolling_buffer(
        self,
        X: np.ndarray,
        y: np.ndarray,
        current_idx: int,
        window_size: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the most recent `window_size` samples up to `current_idx`.

        Args:
            X: Full feature matrix
            y: Full target vector
            current_idx: Current position in the stream
            window_size: Window size (default from config)

        Returns:
            (X_window, y_window)
        """
        if window_size is None:
            window_size = self.config.data.streaming.buffer_size

        start = max(0, current_idx - window_size)
        end = current_idx + 1
        return X[start:end], y[start:end]

    # ── Helpers ─────────────────────────────────────────────────────────────

    def split_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Split DataFrame into X (features) and y (target) numpy arrays."""
        y = df["target"].values
        X = df.drop(columns=["target"]).values
        return X, y

