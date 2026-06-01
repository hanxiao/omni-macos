# Omni - common tasks.
# The generated Omni.xcodeproj shadows the SwiftPM package for xcodebuild, so the
# test target temporarily moves it aside.

DEST = platform=macOS
MODEL ?= /private/tmp/omni-model

.PHONY: app test fixtures clean

# Generate the Xcode project and build the app.
app:
	xcodegen generate
	xcodebuild -project Omni.xcodeproj -scheme Omni -destination '$(DEST)' build

# Run the numeric parity + end-to-end search tests (compiles the Metal shaders).
test:
	@if [ -d Omni.xcodeproj ]; then mv Omni.xcodeproj /tmp/Omni.xcodeproj.bak; fi
	OMNI_MODEL_DIR='$(MODEL)' xcodebuild test -scheme Omni-Package -destination '$(DEST)' -only-testing:OmniKitTests; \
	status=$$?; \
	if [ -d /tmp/Omni.xcodeproj.bak ]; then mv /tmp/Omni.xcodeproj.bak Omni.xcodeproj; fi; \
	exit $$status

# Regenerate Python reference fixtures (run in an env with mlx + tokenizers).
fixtures:
	uv run python Tools/gen_fixtures.py
	cp Fixtures/text_fixtures.json Tests/OmniKitTests/Resources/

clean:
	rm -rf .build Omni.xcodeproj
