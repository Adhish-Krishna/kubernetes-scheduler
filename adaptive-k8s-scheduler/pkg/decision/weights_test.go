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
		p.WeightIdle +
		p.WeightReplica +
		p.WeightPriority +
		p.WeightState +
		p.WeightPDB +
		p.WeightCheckpoint

	if math.Abs(sum-1.0) > 1e-6 {
		t.Fatalf("DefaultPolicy weights must sum to 1.0, got: %.9f", sum)
	}

	// Verify all weights are positive
	weights := []float64{
		p.WeightCPU, p.WeightMemory, p.WeightBenefit, p.WeightIdle,
		p.WeightReplica, p.WeightPriority, p.WeightState, p.WeightPDB, p.WeightCheckpoint,
	}
	for i, w := range weights {
		if w <= 0 {
			t.Errorf("Weight at index %d is not positive: %f", i, w)
		}
	}
}

func TestLoadPolicyFromWeightsFile(t *testing.T) {
	validJSON := `{
		"metadata": {"sample_size": 1000},
		"weights": {
			"WeightCPU": 0.183617150,
			"WeightMemory": 0.259222924,
			"WeightBenefit": 0.251511995,
			"WeightIdle": 0.071811839,
			"WeightReplica": 0.005768108,
			"WeightPriority": 0.056780330,
			"WeightState": 0.117221444,
			"WeightPDB": 0.043597250,
			"WeightCheckpoint": 0.010468960
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

	if math.Abs(p.WeightCPU-0.183617150) > 1e-6 {
		t.Errorf("Expected WeightCPU ~0.183617150, got: %f", p.WeightCPU)
	}
	if math.Abs(p.WeightMemory-0.259222924) > 1e-6 {
		t.Errorf("Expected WeightMemory ~0.259222924, got: %f", p.WeightMemory)
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
