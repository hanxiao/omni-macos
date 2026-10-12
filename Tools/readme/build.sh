#!/bin/bash
# Renders appshot.tex to docs/assets/appshot.png (white background, 2x for retina READMEs).
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$out" appshot.tex >/dev/null
pdftoppm -png -r 300 -singlefile "$out/appshot.pdf" "$out/appshot"
magick "$out/appshot.png" -background white -alpha remove -alpha off -strip ../../docs/assets/appshot.png
rm -rf "$out"
magick identify ../../docs/assets/appshot.png
