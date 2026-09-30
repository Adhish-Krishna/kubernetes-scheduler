package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"

	"simulator/backend/cluster"
	"simulator/backend/models"
	"simulator/backend/simulation"
)

// ClusterAPIHandler provides HTTP handlers for the Active Workload / Cluster mode.
type ClusterAPIHandler struct {
	runner     *simulation.PipelineRunner
	discoverer *cluster.Discoverer
	bridge     *cluster.MetricsBridge
	criuMgr    *cluster.CRIUManager
	stateStore *cluster.StateStore
	lifecycle  *cluster.LifecycleCoordinator
	configPath string
	logger     *zap.Logger
}

// NewClusterAPIHandler constructs the cluster handler with all sub-components.
func NewClusterAPIHandler(runner *simulation.PipelineRunner, configPath string) *ClusterAPIHandler {
	logger, _ := zap.NewProduction()
	if logger == nil {
		logger = zap.NewNop()
	}

	h := &ClusterAPIHandler{
		runner:     runner,
		stateStore: cluster.NewStateStore(),
		configPath: configPath,
		logger:     logger,
	}

	if h.configPath == "" {
		candidates := []string{
			"config/reclaim_policy.json",
			"simulator/config/reclaim_policy.json",
			"../config/reclaim_policy.json",
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				h.configPath = c
				break
			}
		}
	}

	if h.configPath != "" {
		cfg, err := cluster.LoadReclaimConfig(h.configPath)
		if err == nil {
			runner.UpdatePolicy(cfg.ToPolicy())
			logger.Info("Loaded reclaim_policy.json into pipeline", zap.String("path", h.configPath))
		} else {
			logger.Warn("Could not load reclaim_policy.json, using default policy", zap.Error(err))
		}
	}

	disc, err := cluster.NewDiscoverer("")
	if err == nil {
		h.discoverer = disc
		h.criuMgr = cluster.NewCRIUManager(disc.Client(), disc.RESTConfig(), logger)
		h.lifecycle = cluster.NewLifecycleCoordinator(disc.Client(), h.criuMgr, h.stateStore, logger)
	} else {
		logger.Warn("Kubernetes cluster not available; cluster endpoints will return disconnected status", zap.Error(err))
	}

	promURL := os.Getenv("PROMETHEUS_URL")
	if promURL == "" {
		promURL = "http://127.0.0.1:9090"
	}
	bridge, err := cluster.NewMetricsBridge(promURL, logger)
	if err == nil {
		h.bridge = bridge
	} else {
		logger.Warn("Prometheus not available; workload metrics will use zero values", zap.Error(err))
	}

	return h
}

// HandleClusterStatus returns cluster connectivity, CRIU health, and Prometheus status.
func (h *ClusterAPIHandler) HandleClusterStatus(w http.ResponseWriter, r *http.Request) {
	EnableCORS(w)
	if r.Method == http.MethodOptions {
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	resp := map[string]interface{}{
		"mode":      "ACTIVE_CLUSTER",
		"queriedAt": time.Now().UTC().Format(time.RFC3339),
	}

	if h.discoverer != nil {
		summary, err := h.discoverer.GetClusterSummary(ctx)
		if err == nil {
			for k, v := range summary {
				resp[k] = v
			}
		} else {
			resp["status"] = "ERROR"
			resp["error"] = err.Error()
		}
	} else {
		resp["status"] = "DISCONNECTED"
	}

	if h.criuMgr != nil {
		resp["criu"] = h.criuMgr.CheckStatus(ctx)
	}

	if h.bridge != nil {
		resp["prometheus"] = h.bridge.Status(ctx)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// HandleWorkloads discovers real pods, enriches them with Prometheus metrics, and runs
// each through the existing pipeline.
func (h *ClusterAPIHandler) HandleWorkloads(w http.ResponseWriter, r *http.Request) {
	EnableCORS(w)
	if r.Method == http.MethodOptions {
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	if h.discoverer == nil {
		http.Error(w, `{"error":"Kubernetes cluster not connected"}`, http.StatusServiceUnavailable)
		return
	}

	namespace := r.URL.Query().Get("namespace")
	if namespace == "" {
		namespace = "ecommerce"
	}

	workloads, err := h.discoverer.DiscoverWorkloads(ctx, namespace)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"Discovery failed: %s"}`, err.Error()), http.StatusInternalServerError)
		return
	}

	if h.bridge != nil {
		workloads, _ = h.bridge.EnrichWorkloads(ctx, workloads)
	}

	type WorkloadClusterResponse struct {
		Simulation models.WorkloadSimulationResult `json:"simulation"`
		Lifecycle  *cluster.WorkloadClusterState   `json:"lifecycle"`
	}

	results := make([]WorkloadClusterResponse, 0, len(workloads))
	for _, sw := range workloads {
		simResult := h.runner.ExecuteSingleWorkload(sw)

		safetyPassed := simResult.Capabilities.FullReclaimAllowed || simResult.Capabilities.SoftReclaimAllowed
		h.stateStore.RecordDecision(sw.Namespace, sw.Name, simResult.Score, simResult.Action.String(), safetyPassed, simResult.DecisionReasons)

		lifecycle := h.stateStore.Get(sw.Namespace, sw.Name)

		results = append(results, WorkloadClusterResponse{
			Simulation: simResult,
			Lifecycle:  lifecycle,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(results)
}

// HandleReclaimConfig handles GET (read) and PUT (update) of reclaim_policy.json.
func (h *ClusterAPIHandler) HandleReclaimConfig(w http.ResponseWriter, r *http.Request) {
	EnableCORS(w)
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
		return
	}

	switch r.Method {
	case http.MethodGet:
		h.getReclaimConfig(w)
	case http.MethodPut:
		h.putReclaimConfig(w, r)
	default:
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
	}
}

func (h *ClusterAPIHandler) getReclaimConfig(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")

	if h.configPath != "" {
		cfg, err := cluster.LoadReclaimConfig(h.configPath)
		if err == nil {
			_ = json.NewEncoder(w).Encode(cfg)
			return
		}
	}

	_ = json.NewEncoder(w).Encode(cluster.DefaultReclaimConfig())
}

func (h *ClusterAPIHandler) putReclaimConfig(w http.ResponseWriter, r *http.Request) {
	var cfg cluster.ReclaimConfig
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %s"}`, err.Error()), http.StatusBadRequest)
		return
	}

	if err := cfg.Validate(); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"validation failed: %s"}`, err.Error()), http.StatusBadRequest)
		return
	}

	policy := cfg.ToPolicy()
	h.runner.UpdatePolicy(policy)

	if h.configPath != "" {
		data, err := json.MarshalIndent(&cfg, "", "  ")
		if err == nil {
			_ = os.MkdirAll(filepath.Dir(h.configPath), 0755)
			_ = os.WriteFile(h.configPath, data, 0644)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Reclaim policy updated and applied to pipeline",
		"config":  cfg,
	})
}

// HandleCheckpoint triggers a real CRIU checkpoint + reclamation for a workload.
func (h *ClusterAPIHandler) HandleCheckpoint(w http.ResponseWriter, r *http.Request) {
	EnableCORS(w)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Namespace string `json:"namespace"`
		Name      string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %s"}`, err.Error()), http.StatusBadRequest)
		return
	}
	if req.Namespace == "" || req.Name == "" {
		http.Error(w, `{"error":"namespace and name are required"}`, http.StatusBadRequest)
		return
	}

	if h.lifecycle == nil {
		http.Error(w, `{"error":"Kubernetes cluster not connected — cannot checkpoint"}`, http.StatusServiceUnavailable)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	state, err := h.lifecycle.ExecuteReclamation(ctx, req.Namespace, req.Name)
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   err.Error(),
			"state":   state,
		})
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"state":   state,
	})
}

// HandleRestore triggers reconstitution of a reclaimed workload.
func (h *ClusterAPIHandler) HandleRestore(w http.ResponseWriter, r *http.Request) {
	EnableCORS(w)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Namespace string `json:"namespace"`
		Name      string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %s"}`, err.Error()), http.StatusBadRequest)
		return
	}
	if req.Namespace == "" || req.Name == "" {
		http.Error(w, `{"error":"namespace and name are required"}`, http.StatusBadRequest)
		return
	}

	if h.lifecycle == nil {
		http.Error(w, `{"error":"Kubernetes cluster not connected — cannot restore"}`, http.StatusServiceUnavailable)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	state, err := h.lifecycle.ExecuteRestore(ctx, req.Namespace, req.Name)
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   err.Error(),
			"state":   state,
		})
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"state":   state,
	})
}
