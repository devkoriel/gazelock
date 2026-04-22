.PHONY: all generate build build-release test test-unit lint lint-fix clean run verify setup help

PROJECT_NAME = GazeLock
SCHEME = GazeLock
CONFIG_DEBUG = Debug
CONFIG_RELEASE = Release
BUILD_DIR = build
DERIVED_DATA = $(BUILD_DIR)/DerivedData

XCODEBUILD = xcodebuild
XCODEBUILD_FLAGS = -project $(PROJECT_NAME).xcodeproj -scheme $(SCHEME) -derivedDataPath $(DERIVED_DATA)
XCPRETTY ?= xcpretty

all: generate build

generate:
	@echo "==> Generating Xcode project from project.yml"
	@xcodegen generate
	@echo "    Done: $(PROJECT_NAME).xcodeproj"

build: generate
	@echo "==> Building $(PROJECT_NAME) (Debug)"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) build

build-release: generate
	@echo "==> Building $(PROJECT_NAME) (Release)"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_RELEASE) build

test: generate
	@echo "==> Running tests"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) test

test-unit: generate
	@echo "==> Running unit tests"
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) -configuration $(CONFIG_DEBUG) \
		-only-testing:GazeLockTests test

lint:
	@echo "==> Linting with SwiftLint"
	@swiftlint lint --config .swiftlint.yml --quiet

lint-fix:
	@echo "==> Auto-fixing lint issues"
	@swiftlint lint --fix --config .swiftlint.yml --quiet

clean:
	@echo "==> Cleaning build artifacts"
	@rm -rf $(BUILD_DIR)
	@rm -rf $(PROJECT_NAME).xcodeproj
	$(XCODEBUILD) $(XCODEBUILD_FLAGS) clean 2>/dev/null || true

run: build
	@echo "==> Launching $(PROJECT_NAME)"
	@open "$(DERIVED_DATA)/Build/Products/$(CONFIG_DEBUG)/$(PROJECT_NAME).app"

verify: lint test
	@echo "==> verify: lint + test passed"

setup:
	@echo "==> Installing dev tools"
	@which xcodegen > /dev/null || brew install xcodegen
	@which swiftlint > /dev/null || brew install swiftlint
	@which create-dmg > /dev/null || brew install create-dmg
	@echo "    Done. Remember to export DEVELOPMENT_TEAM."

help:
	@echo "GazeLock Makefile — available targets:"
	@echo "  make              -> generate + build (Debug)"
	@echo "  make generate     -> xcodegen generate"
	@echo "  make build        -> Debug build"
	@echo "  make build-release-> Release build"
	@echo "  make test         -> All tests"
	@echo "  make test-unit    -> Unit tests only"
	@echo "  make lint         -> SwiftLint"
	@echo "  make lint-fix     -> SwiftLint auto-fix"
	@echo "  make verify       -> lint + test (CI parity)"
	@echo "  make clean        -> Remove build artifacts + generated project"
	@echo "  make run          -> Build and open the app"
	@echo "  make setup        -> Install dev tools via Homebrew"
