#!/bin/bash

echo "📦 Starting Ollama installation using Homebrew..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null
then
    echo "❌ Homebrew is not installed. Please install it first: https://brew.sh/"
    exit 1
fi

# Install Ollama via Homebrew
echo "🚀 Installing Ollama with brew..."
brew install --cask ollama

# Check if installation was successful
if command -v ollama &> /dev/null
then
    echo "✅ Ollama was installed successfully."
    echo "👉 You can now run: ollama run llama3"
else
    echo "❌ Ollama installation failed."
fi