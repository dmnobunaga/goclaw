package providers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestResilientOpenAIProvider_RetryAcrossBackends(t *testing.T) {
	var c1 int32
	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&c1, 1)
		// Simulate rate limit
		w.Header().Set("Retry-After", "1")
		w.WriteHeader(429)
		_, _ = w.Write([]byte("rate limited"))
	}))
	defer srv1.Close()

	var c2 int32
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&c2, 1)
		w.WriteHeader(503)
		_, _ = w.Write([]byte("service unavailable"))
	}))
	defer srv2.Close()

	srv3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return a minimal OpenAI-like JSON body
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv3.Close()

	policy := ResilientRetryPolicy{
		MaxAttempts:       6,
		MaxElapsed:        10 * time.Second,
		MinDelay429:       10 * time.Millisecond,
		MaxDelay429:       100 * time.Millisecond,
		MinDelay5xx:       10 * time.Millisecond,
		MaxDelay5xx:       100 * time.Millisecond,
		MinDelayNetwork:   5 * time.Millisecond,
		MaxDelayNetwork:   100 * time.Millisecond,
		Jitter:            0.05,
		RespectRetryAfter: true,
	}

	backends := []BackendConfig{
		{Name: "b1", APIBase: srv1.URL, DefaultModel: "m"},
		{Name: "b2", APIBase: srv2.URL, DefaultModel: "m"},
		{Name: "b3", APIBase: srv3.URL, DefaultModel: "m"},
	}

	p := NewResilientOpenAIProvider("res", backends, policy)

	// Ensure providers use test server clients so connections succeed deterministically
	p.backends[0].provider.client = srv1.Client()
	p.backends[1].provider.client = srv2.Client()
	p.backends[2].provider.client = srv3.Client()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}}

	resp, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}
	if resp == nil || resp.Content != "hello" {
		t.Fatalf("unexpected response: %#v", resp)
	}

	if atomic.LoadInt32(&c1) == 0 {
		t.Fatalf("backend1 was not called")
	}
	if atomic.LoadInt32(&c2) == 0 {
		t.Fatalf("backend2 was not called")
	}

	// backend1 and backend2 should have recorded failures
	if p.backends[0].failureCount == 0 {
		t.Fatalf("expected backend1 to have failures recorded")
	}
	if p.backends[1].failureCount == 0 {
		t.Fatalf("expected backend2 to have failures recorded")
	}
}
