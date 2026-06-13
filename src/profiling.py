import time
import torch


class ProfilingHarness:
    """
    Latency and memory profiler for the ShieldEngine.
    """
    def __init__(self, model, shield, scenario: str = "cloud", warmup: int = 3):
        self.model    = model
        self.shield   = shield
        self.scenario = scenario
        self.warmup   = warmup
        self.device   = shield.device

    def measure_latency_and_memory(self, input_batch: dict) -> dict:
        batch = {k: v.to(self.device) for k, v in input_batch.items()} \
                if hasattr(input_batch, "items") else input_batch
                
        # Populate activation buffer via forward pass FIRST (not part of Shield overhead)
        with torch.no_grad():
            _ = self.model(**batch)

        # ── Warmup (eliminates cold-start CUDA JIT overhead) ──────────────
        if self.warmup > 0:
            for _ in range(self.warmup):
                _ = self.shield.get_anomaly_score()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # ── Memory baseline ───────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            mem_before = torch.cuda.memory_allocated(self.device)

        # ── Timing (Strictly Shield Overhead) ─────────────────────────────
        if torch.cuda.is_available():
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            start_ev.record()

            with torch.no_grad():
                score    = self.shield.get_anomaly_score()
                detected = bool(score > self.shield.threshold) if self.shield._fitted else False

            end_ev.record()
            torch.cuda.synchronize()
            latency_ms = start_ev.elapsed_time(end_ev)   # true GPU wall time

        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                score    = self.shield.get_anomaly_score()
                detected = bool(score > self.shield.threshold) if self.shield._fitted else False
            latency_ms = (time.perf_counter() - t0) * 1000.0

        # ── Memory delta ──────────────────────────────────────────────────
        if torch.cuda.is_available():
            mem_peak  = torch.cuda.max_memory_allocated(self.device)
            mem_overhead_mb = (mem_peak - mem_before) / (1024 ** 2)
        else:
            mem_overhead_mb = 0.0

        return {
            "latency_ms":         latency_ms,
            "memory_overhead_mb": mem_overhead_mb,
            "detected_attack":    detected,
            "anomaly_score":      score,
        }

    def run_throughput_benchmark(self, input_batch: dict, n_calls: int = 100) -> dict:
        batch = {k: v.to(self.device) for k, v in input_batch.items()} \
                if hasattr(input_batch, "items") else input_batch

        # Populate buffer once
        with torch.no_grad():
            _ = self.model(**batch)

        latencies = []
        for _ in range(n_calls):
            if torch.cuda.is_available():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                with torch.no_grad():
                    _ = self.shield.get_anomaly_score()
                e.record()
                torch.cuda.synchronize()
                latencies.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = self.shield.get_anomaly_score()
                latencies.append((time.perf_counter() - t0) * 1000.0)

        import statistics
        latencies_sorted = sorted(latencies)
        mean_ms = statistics.mean(latencies)
        p95_ms  = latencies_sorted[int(0.95 * n_calls)]
        p99_ms  = latencies_sorted[int(0.99 * n_calls)]
        rps     = 1000.0 / mean_ms if mean_ms > 0 else float('inf')

        print(f"\n[Profiler] Throughput benchmark ({n_calls} calls):")
        print(f"  Mean latency : {mean_ms:.2f} ms")
        print(f"  P95  latency : {p95_ms:.2f} ms")
        print(f"  P99  latency : {p99_ms:.2f} ms")
        print(f"  Throughput   : {rps:.1f} req/s")

        return {
            "mean_latency_ms": mean_ms,
            "p95_latency_ms":  p95_ms,
            "p99_latency_ms":  p99_ms,
            "throughput_rps":  rps,
        }
