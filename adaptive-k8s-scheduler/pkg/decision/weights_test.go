package decision

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultPolicyWeightsSum(t *testing.T) {
	p := DefaultPolicy()
	sum := p.WeightCPU +
		p.WeightMemory +
		p.WeightBenefit +
		p.WeightPriority +
		p.WeightState +
		p.WeightReplica

	if math.Abs(sum-1.0) > 1e-5 {
		t.Fatalf("DefaultPolicy 6 weights must sum to 1.0, got: %.6f", sum)
	}

	// Verify all 6 active weights are positive
	weights := []struct {
		name string
		val  float64
	}{
		{"WeightCPU", p.WeightCPU},
		{"WeightMemory", p.WeightMemory},
		{"WeightBenefit", p.WeightBenefit},
		{"WeightPriority", p.WeightPriority},
		{"WeightState", p.WeightState},
		{"WeightReplica", p.WeightReplica},
	}
	for _, w := range weights {
		if w.val <= 0 {
			t.Errorf("%s is not positive: %f", w.name, w.val)
		}
	}

	// Verify data-backed thresholds
	if p.SoftReclaimScoreThreshold <= 0 || p.SoftReclaimScoreThreshold >= p.FullReclaimScoreThreshold {
		t.Errorf("Invalid threshold ordering: Soft=%.4f, Full=%.4f", p.SoftReclaimScoreThreshold, p.FullReclaimScoreThreshold)
	}
}

func TestLoadPolicyFromWeightsFile(t *testing.T) {
	validJSON := `{
		"metadata": {"sample_size": 405120},
		"weights": {
			"WeightCPU": 0.199702,
			"WeightMemory": 0.225841,
			"WeightBenefit": 0.409373,
			"WeightPriority": 0.063951,
			"WeightState": 0.098443,
			"WeightReplica": 0.002690
		},
		"thresholds": {
			"SoftReclaimScoreThreshold": 0.3361,
			"FullReclaimScoreThreshold": 0.5927
		}
	}`

	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "trained_weights.json")
	if err := os.WriteFile(filePath, []byte(validJSON), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	p, err := LoadPolicyFromWeightsFile(filePath)
	if err != nil {
		t.Fatalf("Expected successful load, got error: %v", err)
	}

	if math.Abs(p.WeightCPU-0.199702) > 1e-6 {
		t.Errorf("Expected WeightCPU ~0.199702, got: %f", p.WeightCPU)
	}
	if math.Abs(p.WeightBenefit-0.409373) > 1e-6 {
		t.Errorf("Expected WeightBenefit ~0.409373, got: %f", p.WeightBenefit)
	}
	if math.Abs(p.SoftReclaimScoreThreshold-0.3361) > 1e-6 {
		t.Errorf("Expected SoftThreshold ~0.3361, got: %f", p.SoftReclaimScoreThreshold)
	}
	if math.Abs(p.FullReclaimScoreThreshold-0.5927) > 1e-6 {
		t.Errorf("Expected FullThreshold ~0.5927, got: %f", p.FullReclaimScoreThreshold)
	}
}

func TestLoadPolicyInvalidWeights(t *testing.T) {
	invalidSumJSON := `{
		"weights": {
			"WeightCPU": 0.9,
			"WeightMemory": 0.9
		}
	}`

	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "invalid.json")
	if err := os.WriteFile(filePath, []byte(invalidSumJSON), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	_, err := LoadPolicyFromWeightsFile(filePath)
	if err == nil {
		t.Fatal("Expected error on invalid weights sum, got nil")
	}
}
