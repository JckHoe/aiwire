GO ?= $(shell which go 2>/dev/null)

.PHONY: all help build clean test test-race vet fmt \
	test-openai test-openrouter test-zai test-cerebras test-integration test-usage

all: clean build test

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Targets:"
	@echo "  all              Clean, build, and run unit tests"
	@echo "  build            Compile the library (go build ./...)"
	@echo "  vet              Run go vet"
	@echo "  fmt              Run gofmt -w on all sources"
	@echo "  test             Run unit tests (no network)"
	@echo "  test-race        Run unit tests with the race detector"
	@echo "  test-openai      Run integration tests against OpenAI       (needs OPENAI_API_KEY)"
	@echo "  test-openrouter  Run integration tests against OpenRouter   (needs OPENROUTER_API_KEY)"
	@echo "  test-zai         Run integration tests against Z.ai         (needs ZAI_API_KEY)"
	@echo "  test-cerebras    Run integration tests against Cerebras    (needs CEREBRAS_API_KEY)"
	@echo "  test-usage       Run usage/cache-token tests                (needs OPENAI_API_KEY and OPENROUTER_API_KEY)"
	@echo "  test-integration Run all integration tests"
	@echo "  clean            Clear the Go test cache"

build:
	$(GO) build ./...

vet:
	$(GO) vet ./...

fmt:
	$(GO) fmt ./...

clean:
	$(GO) clean -testcache

test:
	$(GO) test -v ./...

test-race:
	$(GO) test -v -race ./...

test-openai:
	$(GO) test -v -tags=openai -count=1 ./...

test-openrouter:
	$(GO) test -v -tags=openrouter -count=1 ./...

test-zai:
	$(GO) test -v -tags=zai -count=1 ./...

test-cerebras:
	$(GO) test -v -tags=cerebras -count=1 ./...

test-usage:
	$(GO) test -v -tags=usage -count=1 ./...

test-integration: test-openai test-openrouter test-zai test-cerebras
