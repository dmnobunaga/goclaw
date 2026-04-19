package providers

import (
	"context"
	"reflect"
	"testing"
	"time"
)

func TestPickAvailableBackendRoundRobin(t *testing.T) {
	backends := []BackendConfig{{Name: "b1"}, {Name: "b2"}, {Name: "b3"}}
	p := NewResilientOpenAIProvider("res", backends, ResilientRetryPolicy{})
	p.mu.Lock()
	p.nextIndex = 1
	p.mu.Unlock()

	got := []string{}
	for i := 0; i < 4; i++ {
		b, wait := p.pickAvailableBackend(time.Now())
		if wait != 0 {
			t.Fatalf("unexpected wait: %v", wait)
		}
		if b == nil {
			t.Fatalf("expected backend, got nil")
		}
		got = append(got, b.cfg.Name)
	}

	want := []string{"b2", "b3", "b1", "b2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round-robin mismatch: got %v want %v", got, want)
	}
}

func TestWaitDoesNotConsumeAttempts(t *testing.T) {
	backends := []BackendConfig{{Name: "b1"}}
	policy := DefaultResilientRetryPolicy()
	policy.MaxAttempts = 1
	policy.MaxElapsed = 5 * time.Second
	p := NewResilientOpenAIProvider("res", backends, policy)

	// force an initial cooldown so the loop must wait before calling a backend
	p.mu.Lock()
	if len(p.backends) == 0 {
		p.mu.Unlock()
		t.Fatalf("no backends")
	}
	p.backends[0].cooldownUntil = time.Now().Add(200 * time.Millisecond)
	p.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Use the internal doWithRetryChat with a dummy op that succeeds.
	res, err := p.doWithRetryChat(ctx, "test", func(provider *OpenAIProvider) (*ChatResponse, error) {
		return &ChatResponse{Content: "ok", FinishReason: "stop"}, nil
	})
	if err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}
	if res == nil || res.Content != "ok" {
		t.Fatalf("unexpected response: %v", res)
	}
}
