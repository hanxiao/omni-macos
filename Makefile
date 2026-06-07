# Omni - common tasks.

DEST = platform=macOS
MODEL ?= /private/tmp/omni-model
ONLY ?=

.PHONY: app test fixtures clean

# Generate the Xcode project and build the app.
app:
	xcodegen generate
	xcodebuild -project Omni.xcodeproj -scheme Omni -destination '$(DEST)' build

# Run the numeric parity + end-to-end search tests (compiles the Metal shaders). Delegated to
# run-tests.sh, which applies the swift-tokenizers Rust-artifact overrides and ad-hoc signs the
# bundle so xctest can load it (plain `xcodebuild test` fails on both counts in this project).
# Filter a single class with: make test ONLY=OmniKitTests.VectorStoreTests
test:
	OMNI_MODEL_DIR='$(MODEL)' ./Scripts/run-tests.sh $(ONLY)

# Regenerate Python reference fixtures (run in an env with mlx + tokenizers).
fixtures:
	uv run python Tools/gen_fixtures.py
	cp Fixtures/text_fixtures.json Tests/OmniKitTests/Resources/

clean:
	rm -rf .build Omni.xcodeproj
