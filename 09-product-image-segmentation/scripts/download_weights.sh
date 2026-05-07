#!/usr/bin/env bash
# Download model weights to ./weights/
# BiRefNet loads automatically via HuggingFace — no manual download needed.
# This script covers U2Net and MODNet.

set -e
WEIGHTS_DIR="$(dirname "$0")/../weights"
mkdir -p "$WEIGHTS_DIR"

echo "=== Downloading U2Net weights ==="
U2NET_URL="https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
# pip install gdown first
gdown "$U2NET_URL" -O "$WEIGHTS_DIR/u2net.pth" || echo "[WARN] gdown failed. Download manually from https://github.com/xuebinqin/U-2-Net"

echo ""
echo "=== Downloading U2Netp (lightweight) weights ==="
U2NETP_URL="https://drive.google.com/uc?id=1rbSTGKAE-MTkiT8n1wmCRF_XiBBs4Jbi"
gdown "$U2NETP_URL" -O "$WEIGHTS_DIR/u2netp.pth" || echo "[WARN] gdown failed."

echo ""
echo "=== MODNet ==="
echo "Download from https://github.com/ZHKKKe/MODNet/tree/master#pretrained-models"
echo "Place modnet_photographic_portrait_matting.ckpt in $WEIGHTS_DIR/"

echo ""
echo "=== BiRefNet ==="
echo "BiRefNet loads automatically from HuggingFace (zhengpeng7/BiRefNet)."
echo "No manual download required."

echo ""
echo "Done. Weights directory:"
ls -lh "$WEIGHTS_DIR/"
