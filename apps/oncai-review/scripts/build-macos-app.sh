#!/usr/bin/env bash
#
# Assemble a double-clickable macOS .app around the console server binary.
#
# The app isn't a Cocoa app — it's a thin launcher whose only job is to open the
# real (console) `oncai-review` binary in a Terminal window. That Terminal window
# is the running app's persistent handle: it shows the http://localhost address,
# and you quit by clicking Quit in the browser, pressing Ctrl-C, or closing the
# window. This sidesteps the windowed-app problems (vanishing Dock icon, the
# bounce-and-hang when you double-click a running headless app).
#
# Usage: build-macos-app.sh <console-binary> <icon.icns> <version> [out-dir]
set -euo pipefail

BIN="${1:?path to the console binary}"
ICON="${2:?path to icon.icns}"
VERSION="${3:?semantic version}"
OUT="${4:-dist}"

APP="$OUT/oncai-review.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BIN" "$APP/Contents/MacOS/oncai-review"
chmod +x "$APP/Contents/MacOS/oncai-review"
cp "$ICON" "$APP/Contents/Resources/icon.icns"

# The bundle executable: opens the server in a new Terminal window. It first
# clears the download quarantine on the bundle so Terminal can run the inner
# binary without a second Gatekeeper block (the user already approved the .app).
cat > "$APP/Contents/MacOS/launch" <<'LAUNCH'
#!/bin/bash
here="$(cd "$(dirname "$0")" && pwd)"
xattr -cr "$here/../.." 2>/dev/null || true
chmod +x "$here/oncai-review" 2>/dev/null || true
open -a Terminal "$here/oncai-review"
LAUNCH
chmod +x "$APP/Contents/MacOS/launch"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>OncAI Review</string>
  <key>CFBundleDisplayName</key><string>OncAI Review</string>
  <key>CFBundleIdentifier</key><string>com.davidhein.oncai-review</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>launch</string>
  <key>CFBundleIconFile</key><string>icon.icns</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
PLIST

echo "Built $APP"
