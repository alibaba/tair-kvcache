"""
Qwen3.6-Plus Request Latency Predictor Wrapper

Based on README specification from colleague's trained models.
"""
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

class RequestLevelTimePredictor:
    """
    Request-level time predictor using 6 HistGradientBoostingRegressor models.
    
    Model selection:
    - Bucket: 0-32k, 32-256k, 256k+ (based on total_tokens = uncached + cached)
    - Split: zero_hit (cached=0), has_hit (cached>0)
    
    Features: inlen, unc, hit, cr, inflp, infld, punc, dtok, attn
    Target: log1p(first_latency_ms) -> expm1 -> ms -> seconds
    """
    
    def __init__(self, models_dir):
        """
        Load all 6 models from joblib files.
        
        Args:
            models_dir: Path to directory containing qwen_request_latency_*.joblib files
        """
        self.models = {}
        model_files = [
            ("0_32k_zero_hit", "0_32k", "zero_hit"),
            ("0_32k_has_hit", "0_32k", "has_hit"),
            ("32_256k_zero_hit", "32_256k", "zero_hit"),
            ("32_256k_has_hit", "32_256k", "has_hit"),
            ("256kplus_zero_hit", "256kplus", "zero_hit"),
            ("256kplus_has_hit", "256kplus", "has_hit"),
        ]
        
        for filename, bucket, split in model_files:
            path = f"{models_dir}/qwen_request_latency_{filename}.joblib"
            data = joblib.load(path)
            self.models[(bucket, split)] = data["model"]
    
    def _select_model(self, total_tokens, cached_tokens):
        """Select model based on total_tokens bucket and cached_tokens split."""
        # Bucket selection
        if total_tokens < 32000:
            bucket = "0_32k"
        elif total_tokens < 256000:
            bucket = "32_256k"
        else:
            bucket = "256kplus"
        
        # Split selection
        split = "zero_hit" if cached_tokens <= 0 else "has_hit"
        
        return self.models[(bucket, split)]
    
    def _derive_features(self, uncached_tokens, cached_tokens, inflp=0, infld=0, punc=0, dtok=0):
        """Derive all 9 features from simulator inputs."""
        inlen = uncached_tokens + cached_tokens
        unc = uncached_tokens
        hit = cached_tokens
        cr = cached_tokens / max(1, inlen)
        attn = unc * (2 * hit + unc) / 1e6
        
        return [inlen, unc, hit, cr, inflp, infld, punc, dtok, attn]
    
    def predict_request_time(self, uncached_tokens, cached_tokens=0, inflp=0, infld=0, punc=0, dtok=0):
        """
        Predict request latency in seconds.
        
        Args:
            uncached_tokens: Number of uncached tokens (required)
            cached_tokens: Number of cached tokens (default: 0)
            inflp: In-flight prefill requests (default: 0)
            infld: In-flight decode requests (default: 0)
            punc: Pending uncached tokens (default: 0)
            dtok: Decode tokens (default: 0)
        
        Returns:
            Predicted latency in seconds
        """
        total_tokens = uncached_tokens + cached_tokens
        
        # Select model
        model = self._select_model(total_tokens, cached_tokens)
        
        # Derive features
        features = self._derive_features(uncached_tokens, cached_tokens, inflp, infld, punc, dtok)
        
        # Predict (returns log1p(ms))
        log1p_ms = model.predict([features])[0]
        
        # Convert to seconds: expm1(log1p_ms) -> ms -> /1000 -> seconds
        ms = np.expm1(log1p_ms)
        seconds = ms / 1000.0
        
        return seconds

# Test
if __name__ == "__main__":
    models_dir = "/sgl-workspace/claude_workspace/data/predictors/qwen36_plus_hit_binary_base_through_20260609_h04"
    predictor = RequestLevelTimePredictor(models_dir)
    
    # Test cases
    test_cases = [
        (1000, 0),      # Small, no cache
        (1000, 500),    # Small, with cache
        (50000, 0),     # Medium, no cache
        (50000, 10000), # Medium, with cache
        (300000, 0),    # Large, no cache
        (300000, 50000) # Large, with cache
    ]
    
    for uncached, cached in test_cases:
        seconds = predictor.predict_request_time(uncached, cached)
        print(f"uncached={uncached:6d}, cached={cached:6d} -> {seconds*1000:7.1f} ms")
