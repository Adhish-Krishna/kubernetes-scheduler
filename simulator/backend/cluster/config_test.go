package cluster

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadReclaimConfig_Valid(t *testing.T) {
	path := "../../config/reclaim_policy.json"
	if _, err := os.Stat(path); err != nil {
		path = "../../../simulator/config/reclaim_policy.json"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("config file not found at %s: %v", path, err)
	}

	cfg, err := LoadReclaimConfig(path)
	if err != nil {
		t.Fatalf("expected valid config, got: %v", err)
	}

	if cfg.Thresholds.FullReclaim != 0.75 {
		t.Errorf("expected full_reclaim=0.75, got %v", cfg.Thresholds.FullReclaim)
	}
	if cfg.Thresholds.SoftReclaim != 0.50 {
		t.Errorf("expected soft_reclaim=0.50, got %v", cfg.Thresholds.SoftReclaim)
	}

	pol := cfg.ToPolicy()
	if pol.WeightCPU != 0.20 || pol.WeightMemory != 0.20 {
		t.Errorf("policy weights mismatch: CPU=%v, Mem=%v", pol.WeightCPU, pol.WeightMemory)
	}
}

func TestReclaimConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*ReclaimConfig)
		wantErr bool
	}{
		{
			name:    "valid default",
			mutate:  func(c *ReclaimConfig) {},
			wantErr: false,
		},
		{
			name: "negative weight",
			mutate: func(c *ReclaimConfig) {
				c.Weights.CPU = -0.1
				c.Weights.Memory = 0.5
			},
			wantErr: true,
		},
		{
			name: "weights do not sum to 1",
			mutate: func(c *ReclaimConfig) {
				c.Weights.CPU = 0.5
			},
			wantErr: true,
		},
		{
			name: "full reclaim threshold > 1",
			mutate: func(c *ReclaimConfig) {
				c.Thresholds.FullReclaim = 1.2
			},
			wantErr: true,
		},
		{
			name: "soft > full threshold",
			mutate: func(c *ReclaimConfig) {
				c.Thresholds.SoftReclaim = 0.8
				c.Thresholds.FullReclaim = 0.7
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultReclaimConfig()
			tc.mutate(cfg)
			err := cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Fatalf("Validate() err = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestLoadReclaimConfig_MalformedJSON(t *testing.T) {
	tmpDir := t.TempDir()
	badFile := filepath.Join(tmpDir, "bad.json")
	if err := os.WriteFile(badFile, []byte("{ bad json"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadReclaimConfig(badFile)
	if err == nil {
		t.Fatal("expected error for malformed json, got nil")
	}
}
