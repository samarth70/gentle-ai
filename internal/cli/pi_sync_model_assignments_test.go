package cli

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gentleman-programming/gentle-ai/v4/internal/model"
)

func piRoutingStorePath(home string) string {
	return filepath.Join(home, ".pi", "agent", "subagents.json")
}

func readPiRoutingStore(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var store map[string]any
	if err := json.Unmarshal(data, &store); err != nil {
		t.Fatal(err)
	}
	return store
}

func piProfiles(t *testing.T, store map[string]any) map[string]any {
	t.Helper()
	profiles, ok := store["model_profiles"].(map[string]any)
	if !ok {
		t.Fatalf("model_profiles = %T, want object", store["model_profiles"])
	}
	return profiles
}

// TestSyncPiAssignmentWritesRoutingStore proves the pi step projects picker
// assignments into subagents.json model_profiles (#6075) while preserving
// user profiles without an assignment and every other key in the file (#4946).
func TestSyncPiAssignmentWritesRoutingStore(t *testing.T) {
	home := t.TempDir()
	path := piRoutingStorePath(home)
	original := []byte(`{
  "model_profiles": {
    "gentle-ai-worker": {"model": "openai/user-worker", "effort": "high"},
    "review-refuter": {"model": "anthropic/user-refuter", "effort": "medium"},
    "custom-helper": {"model": "user-provider/user-model", "effort": "low"}
  },
  "max_concurrency": 3,
  "timeout_ms": 60000,
  "user_setting": {"keep": true}
}
`)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, original, 0600); err != nil {
		t.Fatal(err)
	}
	var changed []string
	step := piModelAssignmentSyncStep{
		path: path,
		assignments: map[string]model.ModelAssignment{
			"gentle-ai-worker": {ProviderID: "provider-a", ModelID: "model-x", Effort: "low"},
			"review-refuter":   {ProviderID: "provider-a", ModelID: "model-y"},
		},
		changedFiles: &changed,
	}
	if err := step.Run(); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	store := readPiRoutingStore(t, path)
	profiles := piProfiles(t, store)
	worker, ok := profiles["gentle-ai-worker"].(map[string]any)
	if !ok || worker["model"] != "provider-a/model-x" || worker["effort"] != "low" {
		t.Fatalf("gentle-ai-worker profile = %v, want model provider-a/model-x effort low", profiles["gentle-ai-worker"])
	}
	refuter, ok := profiles["review-refuter"].(map[string]any)
	if !ok || refuter["model"] != "provider-a/model-y" {
		t.Fatalf("review-refuter profile = %v, want model provider-a/model-y", profiles["review-refuter"])
	}
	// An omitted effort preserves the profile's own value.
	if refuter["effort"] != "medium" {
		t.Fatalf("review-refuter effort = %v, want medium (preserved)", refuter["effort"])
	}
	helper, ok := profiles["custom-helper"].(map[string]any)
	if !ok || helper["model"] != "user-provider/user-model" || helper["effort"] != "low" {
		t.Fatalf("custom-helper profile changed: %v", profiles["custom-helper"])
	}
	if store["max_concurrency"] != float64(3) || store["timeout_ms"] != float64(60000) {
		t.Fatalf("unrelated top-level keys changed: %v", store)
	}
	if userSetting, ok := store["user_setting"].(map[string]any); !ok || userSetting["keep"] != true {
		t.Fatalf("user_setting changed: %v", store["user_setting"])
	}
	if !containsString(changed, path) {
		t.Fatalf("changed paths %v omit routing store", changed)
	}
	// A second run converges: nothing left to change.
	changed = nil
	if err := step.Run(); err != nil {
		t.Fatalf("second Run() error = %v", err)
	}
	if len(changed) != 0 {
		t.Fatalf("unchanged assignment reported as changed: %v", changed)
	}
}

// TestSyncPiAssignmentCreatesRoutingStore proves the step creates
// subagents.json (including the pi agent directory) when no routing store
// exists yet, which is the reporter's case in #6075.
func TestSyncPiAssignmentCreatesRoutingStore(t *testing.T) {
	home := t.TempDir()
	path := piRoutingStorePath(home)
	var changed []string
	step := piModelAssignmentSyncStep{
		path: path,
		assignments: map[string]model.ModelAssignment{
			"gentle-ai-worker": {ProviderID: "provider-a", ModelID: "model-x", Effort: "high"},
		},
		changedFiles: &changed,
	}
	if err := step.Run(); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	store := readPiRoutingStore(t, path)
	profiles := piProfiles(t, store)
	worker, ok := profiles["gentle-ai-worker"].(map[string]any)
	if !ok || worker["model"] != "provider-a/model-x" || worker["effort"] != "high" {
		t.Fatalf("gentle-ai-worker profile = %v, want model provider-a/model-x effort high", profiles["gentle-ai-worker"])
	}
	if !containsString(changed, path) {
		t.Fatalf("changed paths %v omit created routing store", changed)
	}
}

// TestSyncPiAssignmentRejectsMalformedStore proves the step fails closed on
// a malformed routing store instead of discarding the owner's content.
func TestSyncPiAssignmentRejectsMalformedStore(t *testing.T) {
	home := t.TempDir()
	path := piRoutingStorePath(home)
	original := []byte("{not json")
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, original, 0600); err != nil {
		t.Fatal(err)
	}
	var changed []string
	step := piModelAssignmentSyncStep{
		path: path,
		assignments: map[string]model.ModelAssignment{
			"gentle-ai-worker": {ProviderID: "provider-a", ModelID: "model-x"},
		},
		changedFiles: &changed,
	}
	if err := step.Run(); err == nil || !strings.Contains(err.Error(), "malformed") {
		t.Fatalf("Run() error = %v, want malformed refusal", err)
	}
	if data, err := os.ReadFile(path); err != nil || !bytes.Equal(data, original) {
		t.Fatalf("malformed store modified: %q, %v", data, err)
	}
	if len(changed) != 0 {
		t.Fatalf("rejected sync reported changes: %v", changed)
	}
}

// TestSyncPiAssignmentRejectsSymlink proves the step refuses a leaf symlink
// instead of following it onto a target outside the sync snapshot.
func TestSyncPiAssignmentRejectsSymlink(t *testing.T) {
	home := t.TempDir()
	outside := t.TempDir()
	path := piRoutingStorePath(home)
	target := filepath.Join(outside, "user-routing.json")
	original := []byte(`{"model_profiles": {"custom-helper": {"model": "user-provider/user-model"}}}`)
	if err := os.WriteFile(target, original, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, path); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	var changed []string
	step := piModelAssignmentSyncStep{
		path: path,
		assignments: map[string]model.ModelAssignment{
			"custom-helper": {ProviderID: "provider-a", ModelID: "model-x"},
		},
		changedFiles: &changed,
	}
	if err := step.Run(); err == nil {
		t.Fatal("sync accepted leaf symlink")
	}
	if data, err := os.ReadFile(target); err != nil || !bytes.Equal(data, original) {
		t.Fatalf("symlink target modified: %q, %v", data, err)
	}
	if info, err := os.Lstat(path); err != nil || info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("routing store link replaced: %v, %v", info, err)
	}
	if len(changed) != 0 {
		t.Fatalf("rejected sync reported changes: %v", changed)
	}
}

// TestSyncPiAssignmentSkipsUnassignedAndLegacy proves legacy SDD keys, empty
// assignments, and names outside the picker identities are never written,
// so an ordinary sync cannot resurrect retired assignments.
func TestSyncPiAssignmentSkipsUnassignedAndLegacy(t *testing.T) {
	home := t.TempDir()
	path := piRoutingStorePath(home)
	var changed []string
	step := piModelAssignmentSyncStep{
		path: path,
		assignments: map[string]model.ModelAssignment{
			"sdd-verify":       {ProviderID: "provider-a", ModelID: "model-x"},
			"not-a-real-agent": {ProviderID: "provider-a", ModelID: "model-x"},
			"gentle-ai-worker": {ProviderID: "", ModelID: ""},
		},
		changedFiles: &changed,
	}
	if err := step.Run(); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("routing store created for skipped assignments: %v", err)
	}
	if len(changed) != 0 {
		t.Fatalf("skipped sync reported changes: %v", changed)
	}
}

// TestSyncPiAssignmentProjectsThroughRunSync proves the step is wired into the
// global sync plan for pi and lands in the routing store, and that a workspace
// sync never mutates the home root.
func TestSyncPiAssignmentProjectsThroughRunSync(t *testing.T) {
	home := setupPiSyncHost(t)
	selection := model.Selection{
		Agents: []model.AgentID{model.AgentPi},
		ModelAssignments: map[string]model.ModelAssignment{
			"gentle-ai-worker": {ProviderID: "provider-a", ModelID: "model-x", Effort: "high"},
		},
	}
	changed := runSyncInjectionSteps(t, home, selection)
	path := piRoutingStorePath(home)
	if !containsString(changed, path) {
		t.Fatalf("changed paths %v omit routing store", changed)
	}
	store := readPiRoutingStore(t, path)
	profiles := piProfiles(t, store)
	worker, ok := profiles["gentle-ai-worker"].(map[string]any)
	if !ok || worker["model"] != "provider-a/model-x" || worker["effort"] != "high" {
		t.Fatalf("gentle-ai-worker profile = %v, want model provider-a/model-x effort high", profiles["gentle-ai-worker"])
	}

	// Workspace scope must not plan the step at all.
	rt, err := newSyncRuntimeWithScope(home, selection, ScopeWorkspace)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range rt.stagePlan().Apply {
		if step.ID() == "sync:pi:model-assignments" {
			t.Fatal("workspace sync planned the pi routing store step")
		}
	}
}
