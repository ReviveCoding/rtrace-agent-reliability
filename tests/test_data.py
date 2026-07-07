from rtrace.data import generate_benchmark, validate_benchmark


def test_benchmark_quality_passes():
    result = validate_benchmark(generate_benchmark(17))
    assert result["status"] == "PASS"
    assert result["counts"]["final_hard"] > 0
