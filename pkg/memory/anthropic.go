package memory

import (
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// AnthropicEndpointConfig is the optional capability that lets a
// ConsolidateConfig or ExtractConfig point the Anthropic client somewhere other
// than api.anthropic.com — an Anthropic-compatible proxy, relay or gateway.
//
// It is deliberately separate from those interfaces: adding a method to them
// would break every hand-rolled implementation, while an implementation that
// does not carry a base URL has nothing to say here and keeps the SDK default.
type AnthropicEndpointConfig interface {
	GetAnthropicBaseURL() string
}

// anthropicBaseURL reads the optional base URL out of a consolidation or
// extraction config. Absent capability and empty value are the same answer:
// use the SDK default.
func anthropicBaseURL(cfg any) string {
	if c, ok := cfg.(AnthropicEndpointConfig); ok {
		return normalizeAnthropicBaseURL(c.GetAnthropicBaseURL())
	}
	return ""
}

// normalizeAnthropicBaseURL cleans up the one mistake a relay's documentation
// reliably produces: pasting the endpoint (…/v1/messages) or the versioned form
// (…/v1) where the SDK expects the base it appends /v1/messages to. Both are
// accepted and reduced to the base, since the Messages API always lives under
// /v1/messages on a compatible endpoint.
func normalizeAnthropicBaseURL(raw string) string {
	base := strings.TrimRight(strings.TrimSpace(raw), "/")
	base = strings.TrimSuffix(base, "/v1/messages")
	base = strings.TrimSuffix(base, "/v1")
	return strings.TrimRight(base, "/")
}

// newAnthropicClient builds the Messages client every Anthropic call uses.
// An empty baseURL keeps the SDK's own default (and its ANTHROPIC_BASE_URL
// fallback), so the zero value reproduces the previous behaviour exactly.
func newAnthropicClient(key, baseURL string) anthropic.Client {
	opts := []option.RequestOption{option.WithAPIKey(key)}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	return anthropic.NewClient(opts...)
}
