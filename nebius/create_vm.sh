#!/bin/bash
# Create an RT-core GPU VM on Nebius AI Cloud for this project (Isaac Sim needs RT cores: L40S or
# RTX PRO 6000; H100/H200 render incorrectly). Requires the Nebius CLI, an authenticated profile
# (`nebius profile create`) and a default project. Commands follow docs.nebius.com/compute/quickstart.
#
#   NAME=so101-sim ./nebius/create_vm.sh            # L40S (eu-north1), 1 GPU
#   PLATFORM=gpu-rtx6000 PRESET=1gpu-24vcpu-218gb ./nebius/create_vm.sh   # RTX PRO 6000 (us-central1)
set -euo pipefail
NAME="${NAME:-so101-sim}"
PLATFORM="${PLATFORM:-gpu-l40s-d}"          # gpu-l40s-a / gpu-l40s-d (eu-north1), gpu-rtx6000 (us-central1)
PRESET="${PRESET:-1gpu-16vcpu-96gb}"        # L40S-d presets: 1gpu-16vcpu-96gb ... ; L40S-a: 1gpu-16vcpu-64gb ...
DISK_GB="${DISK_GB:-300}"                   # Isaac Sim + Isaac Lab + GR00T + datasets need ~150 GB
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu24.04-cuda13.0}"
SSH_PUBKEY="${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
VM_USER="${VM_USER:-ubuntu}"

command -v nebius >/dev/null || { echo "install the Nebius CLI first: https://docs.nebius.com/cli/install" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required (sudo apt-get install jq)" >&2; exit 1; }

USER_DATA=$(jq -Rrs '.' <<EOT
#cloud-config
users:
  - name: $VM_USER
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$SSH_PUBKEY")
EOT
)

SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
echo "subnet: $SUBNET_ID"
BOOT_DISK_ID=$(nebius compute disk create \
  --name "${NAME}-boot" \
  --size-gibibytes "$DISK_GB" \
  --type network_ssd \
  --source-image-family-image-family "$IMAGE_FAMILY" \
  --block-size-bytes 4096 \
  --format jsonpath='{.metadata.id}')
echo "boot disk: $BOOT_DISK_ID"
VM_ID=$(nebius compute instance create \
  --name "$NAME" \
  --resources-platform "$PLATFORM" \
  --resources-preset "$PRESET" \
  --boot-disk-existing-disk-id "$BOOT_DISK_ID" \
  --boot-disk-attach-mode READ_WRITE \
  --cloud-init-user-data "$USER_DATA" \
  --network-interfaces "[{\"name\": \"default-subnet\", \"subnet_id\": \"$SUBNET_ID\", \"ip_address\": {}, \"public_ip_address\": {}}]" \
  --format jsonpath='{.metadata.id}')
echo "instance: $VM_ID"
sleep 20
PUBLIC_IP=$(nebius compute instance get-by-name --name "$NAME" --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address | split("/")[0]')
echo "ssh $VM_USER@$PUBLIC_IP"
echo "delete later with: nebius compute instance delete --id $VM_ID && nebius compute disk delete --id $BOOT_DISK_ID"
