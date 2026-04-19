package providers

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"sync"
	"time"
)

// BackendConfig describes a single OpenAI backend instance.
type BackendConfig struct {
	Name         string
	APIKey       string
	APIBase      string
	DefaultModel string
}

type backendState struct {
	cfg           BackendConfig
	provider      *OpenAIProvider
	cooldownUntil time.Time
	failureCount  int
	lastError     error
}

// ResilientRetryPolicy controls retry/backoff behavior for the wrapper.
type ResilientRetryPolicy struct {
	MaxAttempts int
	MaxElapsed  time.Duration

	MinDelay429 time.Duration
	MaxDelay429 time.Duration

	MinDelay5xx time.Duration
	MaxDelay5xx time.Duration

	MinDelayNetwork time.Duration
	MaxDelayNetwork time.Duration

	Jitter            float64
	RespectRetryAfter bool
}

type ResilientOpenAIProvider struct {
	name     string
	backends []*backendState
	policy   ResilientRetryPolicy
	mu       sync.Mutex
	// nextIndex is the next backend index to try (round-robin).
	nextIndex int
}

// DefaultResilientRetryPolicy returns sensible defaults.
func DefaultResilientRetryPolicy() ResilientRetryPolicy {
	return ResilientRetryPolicy{
		MaxAttempts: 50,
		MaxElapsed:  30 * time.Minute,

		MinDelay429: 15 * time.Second,
		MaxDelay429: 5 * time.Minute,

		MinDelay5xx: 2 * time.Second,
		MaxDelay5xx: 30 * time.Second,

		MinDelayNetwork: 1 * time.Second,
		MaxDelayNetwork: 15 * time.Second,

		Jitter:            0.15,
		RespectRetryAfter: true,
	}
}

// NewResilientOpenAIProvider constructs a wrapper around multiple OpenAI backends.
func NewResilientOpenAIProvider(name string, backends []BackendConfig, policy ResilientRetryPolicy) *ResilientOpenAIProvider {
	// Merge provided policy into defaults field-by-field. Start with defaults
	// and override only non-zero (or true for booleans) fields from the caller's policy.
	merged := DefaultResilientRetryPolicy()

	if policy.MaxAttempts != 0 {
		merged.MaxAttempts = policy.MaxAttempts
	}
	if policy.MaxElapsed != 0 {
		merged.MaxElapsed = policy.MaxElapsed
	}

	if policy.MinDelay429 != 0 {
		merged.MinDelay429 = policy.MinDelay429
	}
	if policy.MaxDelay429 != 0 {
		merged.MaxDelay429 = policy.MaxDelay429
	}

	if policy.MinDelay5xx != 0 {
		merged.MinDelay5xx = policy.MinDelay5xx
	}
	if policy.MaxDelay5xx != 0 {
		merged.MaxDelay5xx = policy.MaxDelay5xx
	}

	if policy.MinDelayNetwork != 0 {
		merged.MinDelayNetwork = policy.MinDelayNetwork
	}
	if policy.MaxDelayNetwork != 0 {
		merged.MaxDelayNetwork = policy.MaxDelayNetwork
	}

	if policy.Jitter != 0 {
		merged.Jitter = policy.Jitter
	}
	if policy.RespectRetryAfter {
		merged.RespectRetryAfter = policy.RespectRetryAfter
	}

	p := &ResilientOpenAIProvider{name: name, policy: merged}

	for _, bc := range backends {
		prov := NewOpenAIProvider(bc.Name, bc.APIKey, bc.APIBase, bc.DefaultModel)
		bs := &backendState{cfg: bc, provider: prov}
		p.backends = append(p.backends, bs)
	}

	return p
}

// --- simple accessors (use first backend when available)

func (p *ResilientOpenAIProvider) Name() string { return p.name }

func (p *ResilientOpenAIProvider) DefaultModel() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.backends) == 0 {
		return ""
	}
	if p.backends[0].provider != nil {
		return p.backends[0].provider.DefaultModel()
	}
	return p.backends[0].cfg.DefaultModel
}

func (p *ResilientOpenAIProvider) SupportsThinking() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.backends) == 0 || p.backends[0].provider == nil {
		return false
	}
	return p.backends[0].provider.SupportsThinking()
}

func (p *ResilientOpenAIProvider) APIKey() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.backends) == 0 || p.backends[0].provider == nil {
		return ""
	}
	return p.backends[0].provider.APIKey()
}

func (p *ResilientOpenAIProvider) APIBase() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.backends) == 0 || p.backends[0].provider == nil {
		return ""
	}
	return p.backends[0].provider.APIBase()
}

func (p *ResilientOpenAIProvider) AuthPrefix() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.backends) == 0 || p.backends[0].provider == nil {
		return ""
	}
	return p.backends[0].provider.AuthPrefix()
}

func (p *ResilientOpenAIProvider) ProviderType() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.backends) == 0 || p.backends[0].provider == nil {
		return ""
	}
	return p.backends[0].provider.ProviderType()
}

func (p *ResilientOpenAIProvider) Capabilities() ProviderCapabilities {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.backends) == 0 || p.backends[0].provider == nil {
		return ProviderCapabilities{}
	}
	return p.backends[0].provider.Capabilities()
}

// WithRegistry sets the model registry on all underlying backends.
func (p *ResilientOpenAIProvider) WithRegistry(r ModelRegistry) *ResilientOpenAIProvider {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, b := range p.backends {
		if b != nil && b.provider != nil {
			b.provider.WithRegistry(r)
		}
	}
	return p
}

// --- error helpers

func isRateLimitError(err error) bool {
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		return httpErr.Status == 429
	}
	return false
}

func isServerError(err error) bool {
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		switch httpErr.Status {
		case 500, 502, 503, 504:
			return true
		}
		return false
	}
	return false
}

func isRetryableProviderError(err error) bool {
	if err == nil {
		return false
	}
	// reuse package-wide isNetworkError from error_classify.go
	return isRateLimitError(err) || isServerError(err) || isNetworkError(err)
}

func extractRetryAfter(err error) time.Duration {
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		if httpErr.RetryAfter > 0 {
			return httpErr.RetryAfter
		}
	}
	return 0
}

// --- delay computation

func computeResilientDelay(err error, attempt int, policy ResilientRetryPolicy) time.Duration {
	if attempt < 1 {
		attempt = 1
	}

	// 429 handling (Retry-After support)
	if isRateLimitError(err) {
		if policy.RespectRetryAfter {
			if ra := extractRetryAfter(err); ra > 0 {
				return ra
			}
		}
		return computeExponentialWithJitter(float64(policy.MinDelay429), float64(policy.MaxDelay429), attempt, policy.Jitter)
	}

	if isServerError(err) {
		return computeExponentialWithJitter(float64(policy.MinDelay5xx), float64(policy.MaxDelay5xx), attempt, policy.Jitter)
	}

	if isNetworkError(err) {
		return computeExponentialWithJitter(float64(policy.MinDelayNetwork), float64(policy.MaxDelayNetwork), attempt, policy.Jitter)
	}

	// Fallback
	return computeExponentialWithJitter(float64(policy.MinDelayNetwork), float64(policy.MaxDelayNetwork), attempt, policy.Jitter)
}

func computeExponentialWithJitter(minDelayF, maxDelayF float64, attempt int, jitter float64) time.Duration {
	delay := minDelayF * math.Pow(2, float64(attempt-1))
	if time.Duration(delay) > time.Duration(maxDelayF) {
		delay = maxDelayF
	}
	if jitter > 0 {
		rng := (rand.Float64()*2 - 1) * jitter
		delay = delay * (1 + rng)
	}
	if delay < 0 {
		delay = minDelayF
	}
	return time.Duration(delay)
}

// --- backend selection and cooldown

func (p *ResilientOpenAIProvider) pickAvailableBackend(now time.Time) (*backendState, time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	n := len(p.backends)
	if n == 0 {
		return nil, 0
	}

	var earliest time.Time
	foundAny := false

	// Start scanning from nextIndex for simple round-robin.
	start := p.nextIndex % n
	for i := 0; i < n; i++ {
		idx := (start + i) % n
		b := p.backends[idx]
		if b == nil {
			continue
		}
		if b.cooldownUntil.IsZero() || !b.cooldownUntil.After(now) {
			// advance nextIndex to the following backend for next time
			p.nextIndex = (idx + 1) % n
			return b, 0
		}
		if !foundAny || b.cooldownUntil.Before(earliest) {
			earliest = b.cooldownUntil
			foundAny = true
		}
	}

	if !foundAny {
		return nil, 0
	}
	return nil, earliest.Sub(now)
}

func (p *ResilientOpenAIProvider) markBackendCooldown(b *backendState, delay time.Duration, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if b == nil {
		return
	}
	b.cooldownUntil = time.Now().Add(delay)
	b.failureCount++
	b.lastError = err
}

// --- core retry loop (non-generic for compatibility)

func (p *ResilientOpenAIProvider) doWithRetryChat(ctx context.Context, opName string, fn func(provider *OpenAIProvider) (*ChatResponse, error)) (*ChatResponse, error) {
	var zero *ChatResponse

	maxAttempts := p.policy.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 1
	}

	start := time.Now()
	var lastErr error
	attemptsUsed := 0

	for {
		if ctx.Err() != nil {
			return zero, ctx.Err()
		}
		if p.policy.MaxElapsed > 0 && time.Since(start) > p.policy.MaxElapsed {
			return zero, fmt.Errorf("resilient provider %s: max elapsed exceeded", opName)
		}

		if attemptsUsed >= maxAttempts {
			if lastErr != nil {
				return zero, lastErr
			}
			return zero, fmt.Errorf("resilient provider %s: exhausted attempts", opName)
		}

		now := time.Now()
		b, wait := p.pickAvailableBackend(now)
		if b == nil {
			// Nothing available right now — wait and retry (does not consume an attempt)
			if wait <= 0 {
				wait = time.Second
			}
			if p.policy.MaxElapsed > 0 && time.Since(start)+wait > p.policy.MaxElapsed {
				if lastErr != nil {
					return zero, lastErr
				}
				return zero, fmt.Errorf("resilient provider %s: no backends available and max elapsed exceeded", opName)
			}
			select {
			case <-ctx.Done():
				return zero, ctx.Err()
			case <-time.After(wait):
			}
			continue
		}

		// We will attempt a provider call — increment attemptsUsed now
		attemptsUsed++
		attempt := attemptsUsed

		res, err := fn(b.provider)
		if err == nil {
			// Success — clear failure state for backend
			p.mu.Lock()
			b.failureCount = 0
			b.lastError = nil
			b.cooldownUntil = time.Time{}
			p.mu.Unlock()
			return res, nil
		}

		lastErr = err

		if !isRetryableProviderError(err) {
			return zero, err
		}

		delay := computeResilientDelay(err, attempt, p.policy)
		p.markBackendCooldown(b, delay, err)

		slog.Warn("resilient provider retry",
			"backend", b.cfg.Name,
			"attempt", attempt,
			"max_attempts", p.policy.MaxAttempts,
			"delay", delay,
			"elapsed", time.Since(start),
			"error", err.Error(),
		)

		// continue to next iteration (waiting or picking another backend)
	}
}

// --- Chat / ChatStream implementations

func (p *ResilientOpenAIProvider) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	return p.doWithRetryChat(ctx, "chat", func(provider *OpenAIProvider) (*ChatResponse, error) {
		return provider.Chat(ctx, req)
	})
}

func (p *ResilientOpenAIProvider) ChatStream(ctx context.Context, req ChatRequest, onChunk func(StreamChunk)) (*ChatResponse, error) {
	// Safer streaming: only retry when error happens before any chunk is sent.
	maxAttempts := p.policy.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 1
	}
	start := time.Now()
	var lastErr error
	attemptsUsed := 0

	for {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		if p.policy.MaxElapsed > 0 && time.Since(start) > p.policy.MaxElapsed {
			return nil, fmt.Errorf("resilient provider chatstream: max elapsed exceeded")
		}

		if attemptsUsed >= maxAttempts {
			if lastErr != nil {
				return nil, lastErr
			}
			return nil, fmt.Errorf("resilient provider chatstream: exhausted attempts")
		}

		now := time.Now()
		b, wait := p.pickAvailableBackend(now)
		if b == nil {
			if wait <= 0 {
				wait = time.Second
			}
			if p.policy.MaxElapsed > 0 && time.Since(start)+wait > p.policy.MaxElapsed {
				if lastErr != nil {
					return nil, lastErr
				}
				return nil, fmt.Errorf("resilient provider chatstream: no backends available and max elapsed exceeded")
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
			continue
		}

		started := false
		wrapped := func(sc StreamChunk) {
			if !started {
				started = true
			}
			if onChunk != nil {
				onChunk(sc)
			}
		}

		// We'll call provider — consume an attempt only when invoking the backend
		attemptsUsed++
		attempt := attemptsUsed

		resp, err := b.provider.ChatStream(ctx, req, wrapped)
		if err == nil {
			// success — clear backend failure state
			p.mu.Lock()
			b.failureCount = 0
			b.lastError = nil
			b.cooldownUntil = time.Time{}
			p.mu.Unlock()
			return resp, nil
		}

		lastErr = err

		// If stream already started, don't retry — return error
		if started {
			return nil, err
		}

		if !isRetryableProviderError(err) {
			return nil, err
		}

		delay := computeResilientDelay(err, attempt, p.policy)
		p.markBackendCooldown(b, delay, err)

		slog.Warn("resilient provider retry",
			"backend", b.cfg.Name,
			"attempt", attempt,
			"max_attempts", p.policy.MaxAttempts,
			"delay", delay,
			"elapsed", time.Since(start),
			"error", err.Error(),
		)
		// continue to next iteration
	}
}
