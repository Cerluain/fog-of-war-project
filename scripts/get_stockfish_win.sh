#!/usr/bin/env bash

# Get project root and use the git-ignored stockfish/ folder
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="$PROJECT_ROOT/stockfish"
mkdir -p "$SF_DIR"

EXE_PATH="$SF_DIR/stockfish.exe"

# Download only if stockfish.exe doesn't exist yet on this laptop
if [ ! -f "$EXE_PATH" ]; then
  echo "Downloading Stockfish for Windows..." >&2
  URL="https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.tar"
  
  curl -fL "$URL" -o "$SF_DIR/stockfish.tar"
  tar -xf "$SF_DIR/stockfish.tar" -C "$SF_DIR"
  rm -f "$SF_DIR/stockfish.tar"

  # Find the extracted .exe and move it to stockfish/stockfish.exe
  FOUND_EXE=$(find "$SF_DIR" -type f -name "*stockfish*.exe" | head -n 1)
  if [ -n "$FOUND_EXE" ]; then
    mv "$FOUND_EXE" "$EXE_PATH"
  fi
fi

# Convert to Windows path (C:/...) for Python
if command -v cygpath >/dev/null 2>&1; then
  WIN_PATH="$(cygpath -m "$EXE_PATH")"
else
  WIN_PATH="$EXE_PATH"
fi

export STOCKFISH="$WIN_PATH"
echo "STOCKFISH is set to: $STOCKFISH" >&2
echo "$STOCKFISH"