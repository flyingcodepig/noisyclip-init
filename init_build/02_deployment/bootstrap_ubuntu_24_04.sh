#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" != "--yes" ]]; then
  echo "This script installs Docker Engine and NVIDIA Container Toolkit."
  echo "Review it first, then run: bash $0 --yes"
  exit 2
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify operating system." >&2
  exit 2
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID}" != "ubuntu" ]] || [[ "${VERSION_ID}" != "24.04" && "${VERSION_ID}" != "22.04" ]]; then
  echo "Supported hosts are Ubuntu 24.04 or 22.04; found ${ID} ${VERSION_ID}." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver is missing. Use a cloud GPU image or install the driver first." >&2
  exit 3
fi

driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
if [[ ! "${driver_major}" =~ ^[0-9]+$ ]] || (( driver_major < 570 )); then
  echo "NVIDIA driver 570+ is required for the pinned CUDA 12.8 stack; found ${driver_major}." >&2
  exit 3
fi

if dpkg -l docker.io docker-compose docker-compose-v2 podman-docker containerd runc 2>/dev/null | grep -q '^ii'; then
  echo "Conflicting distro Docker/container packages are installed." >&2
  echo "Resolve them manually using Docker's official Ubuntu guide; this script will not remove packages." >&2
  exit 4
fi

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

if ! command -v docker >/dev/null 2>&1; then
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  arch="$(dpkg --print-architecture)"
  codename="${UBUNTU_CODENAME:-${VERSION_CODENAME}}"
  printf '%s\n' \
    'Types: deb' \
    'URIs: https://download.docker.com/linux/ubuntu' \
    "Suites: ${codename}" \
    'Components: stable' \
    "Architectures: ${arch}" \
    'Signed-By: /etc/apt/keyrings/docker.asc' \
    | sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null

  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
toolkit_version="1.19.1-1"
sudo apt-get install -y \
  "nvidia-container-toolkit=${toolkit_version}" \
  "nvidia-container-toolkit-base=${toolkit_version}" \
  "libnvidia-container-tools=${toolkit_version}" \
  "libnvidia-container1=${toolkit_version}"

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
echo "Host bootstrap completed. Re-login if your administrator adds you to the docker group."

