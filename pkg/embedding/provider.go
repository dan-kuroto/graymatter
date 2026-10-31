// Package embedding provides pluggable vector embedding backends for GrayMatter.
//
// Auto-detection order:
//  1. Ollama (default — fires a HEAD to /api/tags)
//  2. OpenAI API (if OPENAI_API_KEY is set)
//  3. Voyage AI API (if VOYAGE_API_KEY is set)
//  4. Keyword-only fallback (zero network deps)
package embedding

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// maxErrorBodyBytes caps how much of a non-200 response body goes into an
// error message.
//
// The three HTTP providers used to io.ReadAll an error body whole. The Ollama
// URL is user-configurable and the others sit behind whatever proxy the
// network imposes, so a hostile or broken endpoint could answer 500 with an
// endless chunked body and exhaust the memory of whatever is embedding. Eight
// kibibytes is more than any real error message needs.
const maxErrorBodyBytes = 8 << 10

// cacheSize bounds the per-provider embedding cache (see OpenAIProvider and
// VoyageProvider): identical texts are embedded once per process.
const cacheSize = 128

// errorBody reads a bounded prefix of an error response for use in a message.
func errorBody(r io.Reader) string {
	data, err := io.ReadAll(io.LimitReader(r, maxErrorBodyBytes))
	if err != nil && len(data) == 0 {
		return fmt.Sprintf("<unreadable: %v>", err)
	}
	return string(data)
}

// Mode mirrors graymatter.EmbeddingMode to avoid circular imports.
type Mode int

const (
	ModeAuto Mode = iota
	ModeOllama
	ModeAnthropic // Deprecated: alias for ModeVoyage; kept for numeric stability.
	ModeKeyword
	ModeOpenAI
	ModeVoyage
)

// Provider generates float32 vector embeddings from text.
// A nil return from Embed signals keyword-only mode to the caller.
type Provider interface {
	// Embed returns a vector for text. Returns nil if unavailable.
	Embed(ctx context.Context, text string) ([]float32, error)
	// Dimensions returns the embedding dimension (0 for keyword provider).
	Dimensions() int
	// Name is a human-readable identifier used in logs and stats.
	Name() string
}

// Config carries provider configuration from graymatter.Config.
type Config struct {
	Mode            Mode
	OllamaURL       string
	OllamaModel     string
	AnthropicAPIKey string // Deprecated: unused by every mode; kept for struct compatibility.
	VoyageAPIKey    string
	VoyageBaseURL   string // optional; empty keeps the Voyage default endpoint
	OpenAIAPIKey    string
	OpenAIBaseURL   string // optional; empty keeps the OpenAI default endpoint
	OpenAIModel     string // defaults to text-embedding-3-small
}

// AutoDetect selects the best available Provider given cfg.
// It probes network endpoints with a short timeout so startup is fast.
func AutoDetect(cfg Config) Provider {
	switch cfg.Mode {
	case ModeOllama:
		return NewOllama(cfg)
	case ModeAnthropic:
		// Legacy explicit mode: resolves to the same Voyage-backed slot.
		// An Anthropic key cannot reach the embeddings chain (the endpoint it
		// used to target does not exist), so without a Voyage key this is
		// keyword-only — no doomed network calls per Put.
		return voyageOrKeyword(cfg)
	case ModeVoyage:
		return voyageOrKeyword(cfg)
	case ModeOpenAI:
		if cfg.OpenAIAPIKey != "" {
			return NewOpenAI(cfg)
		}
		return NewKeyword()
	case ModeKeyword:
		return NewKeyword()
	default: // ModeAuto
		if ollamaReachable(cfg.OllamaURL) {
			return NewOllama(cfg)
		}
		if cfg.OpenAIAPIKey != "" {
			return NewOpenAI(cfg)
		}
		return voyageOrKeyword(cfg)
	}
}

// voyageOrKeyword returns a Voyage provider when a Voyage API key is set,
// and the keyword fallback otherwise. The embeddings slot never dials with
// a credential that cannot succeed: a guaranteed-401 call on every Put is
// worse than no vector channel at all, because the failure is silent.
func voyageOrKeyword(cfg Config) Provider {
	if cfg.VoyageAPIKey != "" {
		return NewVoyage(cfg)
	}
	return NewKeyword()
}

// embeddingsEndpoint resolves a configured base URL to the full request target
// for an OpenAI-style embeddings API.
//
// The published base URLs (https://api.openai.com/v1, api.voyageai.com/v1)
// carry the version prefix, and OpenAI-compatible servers mount the endpoint
// under whatever prefix they choose (…/v1, …/api/paas/v4, …/openai). So a value
// with a path is authoritative and only gets /embeddings appended, unless the
// caller already passed the full endpoint. A bare origin has no path to carry a
// version, so the conventional /v1 is supplied for it. Empty keeps fallback,
// which is how the zero value reproduces the previous hardcoded behaviour.
func embeddingsEndpoint(base, fallback string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if base == "" {
		return fallback
	}
	if strings.HasSuffix(base, "/embeddings") {
		return base
	}
	rest := base
	if scheme := strings.Index(rest, "://"); scheme >= 0 {
		rest = rest[scheme+3:]
	}
	if strings.Contains(rest, "/") {
		return base + "/embeddings"
	}
	return base + "/v1/embeddings"
}

// ollamaReachable does a fast HEAD probe to check if Ollama is up.
// Only a 200 counts as reachable: a captive portal or proxy answers 404/302
// for unknown paths, and treating that as "Ollama is here" sends every
// embedding into guaranteed-failing calls whose failures Put then swallows.
func ollamaReachable(baseURL string) bool {
	if baseURL == "" {
		return false
	}
	c := &http.Client{Timeout: 500 * time.Millisecond}
	resp, err := c.Get(baseURL + "/api/tags")
	if err != nil {
		return false
	}
	_ = resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
