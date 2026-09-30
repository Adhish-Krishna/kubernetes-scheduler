package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"simulator/backend/simulation"
)

func TestHandleReclaimConfig_GetAndPut(t *testing.T) {
	tmpDir := t.TempDir()
	cfgPath := filepath.Join(tmpDir, "reclaim_policy.json")

	initialJSON := `{
  "thresholds": {"full_reclaim": 0.75, "soft_reclaim": 0.50},
  "weights": {"cpu": 0.20, "memory": 0.20, "idle": 0.15, "benefit": 0.15, "replica": 0.10, "priority": 0.05, "pdb": 0.05, "state": 0.05, "checkpoint": 0.05},
  "normalization": {"idle_max_duration_sec": 60, "benefit_max_cpu_millis": 2000, "benefit_max_mem_bytes": 4294967296},
  "safety": {"max_priority_for_reclaim": 100000, "min_replicas_required": 1}
}`
	if err := os.WriteFile(cfgPath, []byte(initialJSON), 0644); err != nil {
		t.Fatal(err)
	}

	runner := simulation.NewPipelineRunner()
	handler := NewClusterAPIHandler(runner, cfgPath)

	// 1. GET /api/reclaim/config
	getReq := httptest.NewRequest(http.MethodGet, "/api/reclaim/config", nil)
	getW := httptest.NewRecorder()
	handler.HandleReclaimConfig(getW, getReq)

	if getW.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", getW.Code)
	}

	// 2. PUT /api/reclaim/config with valid update
	updatedJSON := `{
  "thresholds": {"full_reclaim": 0.80, "soft_reclaim": 0.55},
  "weights": {"cpu": 0.25, "memory": 0.15, "idle": 0.15, "benefit": 0.15, "replica": 0.10, "priority": 0.05, "pdb": 0.05, "state": 0.05, "checkpoint": 0.05},
  "normalization": {"idle_max_duration_sec": 60, "benefit_max_cpu_millis": 2000, "benefit_max_mem_bytes": 4294967296},
  "safety": {"max_priority_for_reclaim": 100000, "min_replicas_required": 1}
}`
	putReq := httptest.NewRequest(http.MethodPut, "/api/reclaim/config", bytes.NewReader([]byte(updatedJSON)))
	putW := httptest.NewRecorder()
	handler.HandleReclaimConfig(putW, putReq)

	if putW.Code != http.StatusOK {
		t.Fatalf("expected 200 on PUT, got %d: %s", putW.Code, putW.Body.String())
	}

	if runner.Policy().FullReclaimScoreThreshold != 0.80 {
		t.Errorf("expected updated full threshold 0.80, got %v", runner.Policy().FullReclaimScoreThreshold)
	}
	if runner.Policy().WeightCPU != 0.25 {
		t.Errorf("expected updated CPU weight 0.25, got %v", runner.Policy().WeightCPU)
	}
}

func TestHandleClusterStatus(t *testing.T) {
	runner := simulation.NewPipelineRunner()
	handler := NewClusterAPIHandler(runner, "")

	req := httptest.NewRequest(http.MethodGet, "/api/cluster/status", nil)
	w := httptest.NewRecorder()
	handler.HandleClusterStatus(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed decoding status response: %v", err)
	}

	if resp["mode"] != "ACTIVE_CLUSTER" {
		t.Errorf("expected mode ACTIVE_CLUSTER, got %v", resp["mode"])
	}
}
