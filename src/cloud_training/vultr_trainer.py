"""
vultr_trainer.py
================
Vultr cloud VM lifecycle manager for bat chirp CNN training.

Responsibilities
----------------
* **Provision** a Vultr bare-metal or cloud-compute instance with the
  requested GPU plan, upload training data to Vultr Object Storage,
  start the training container, and poll until the job finishes.
* **Destroy** the instance once training completes (called from inside
  the container via ``--destroy``, or externally via ``--launch`` after
  the job finishes).

Vultr Object Storage is S3-compatible, so we use ``boto3`` for data
transfer and ``requests`` for the Vultr REST API.

Usage
-----
Launch a training run::

    python vultr_trainer.py --launch \\
        --manifest /qnap/bats/jr_pipeline/data/bat_crops/manifest.csv \\
        --crops-dir /qnap/bats/jr_pipeline/data/bat_crops \\
        --docker-image registry.example.com/bats-train:latest \\
        --vultr-api-key $VULTR_API_KEY \\
        --obj-access $OBJ_ACCESS \\
        --obj-secret $OBJ_SECRET \\
        --obj-endpoint https://sjc1.vultrobjects.com \\
        --bucket bat-crops

Destroy an existing instance (e.g. from inside the container)::

    python vultr_trainer.py --destroy \\
        --instance-id <uuid> \\
        --vultr-api-key $VULTR_API_KEY

Configuration constants
-----------------------
Edit :data:`DEFAULT_PLAN` and :data:`DEFAULT_REGION` to change the
default GPU plan.  A100 80 GB plans (``vcg-a100-2c-120gb-2vgpu``) are
the target; adjust if Vultr updates their plan IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3
import requests
from botocore.config import Config
from logging_service import LoggingService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Vultr REST API base URL.
VULTR_API_BASE = 'https://api.vultr.com/v2'

#: Default GPU plan — two A100 80 GB vGPUs (120 GB RAM, 8 vCPUs).
#: Check https://www.vultr.com/products/cloud-gpu/ for current plan IDs.
DEFAULT_PLAN = 'vcg-a100-2c-120gb-2vgpu'

#: Datacenter region.  sjc = Silicon Valley; closest to Stanford/Bay Area.
DEFAULT_REGION = 'sjc'

#: Ubuntu 22.04 LTS with NVIDIA GPU drivers pre-installed.
#: Retrieve the current OS ID with:
#:   curl -s "https://api.vultr.com/v2/os" | python3 -m json.tool | grep -i ubuntu
DEFAULT_OS_ID = 1743   # Ubuntu 22.04 x64 (check and update if stale)

#: S3 bucket where spectrogram crops and the manifest are stored.
DEFAULT_BUCKET = 'bat-crops'

#: Prefix inside the bucket for the crops tree.
CROPS_PREFIX = 'crops/'

#: Object key for the manifest CSV.
MANIFEST_KEY = 'manifest.csv'

#: Prefix under which model outputs are uploaded.
MODEL_OUT_PREFIX = 'models/efficientnet_b0_v2'

#: How often (seconds) to poll for instance-ready status.
POLL_INTERVAL = 30

#: Maximum time (seconds) to wait for instance to reach ``active`` state.
LAUNCH_TIMEOUT = 600

# ---------------------------------------------------------------------------
# Class VultrTrainer
# ---------------------------------------------------------------------------


class VultrTrainer:
    """
    Manage the full lifecycle of a Vultr GPU instance for CNN training.

    :param vultr_api_key: Vultr API key.
    :param obj_endpoint:  Vultr Object Storage endpoint URL,
                          e.g. ``https://sjc1.vultrobjects.com``.
    :param obj_access:    S3-compatible access key for Object Storage.
    :param obj_secret:    S3-compatible secret key for Object Storage.
    :param bucket:        Object Storage bucket name.
    :param docker_image:  Full Docker image reference for the training
                          container, e.g.
                          ``registry.example.com/bats-train:latest``.
    :param plan:          Vultr plan ID.
    :param region:        Vultr datacenter region code.
    :param os_id:         Vultr OS ID for the instance OS.
    :param dry_run:       If True, log actions but do not call the API.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        vultr_api_key: str,
        obj_endpoint:  str,
        obj_access:    str,
        obj_secret:    str,
        bucket:        str  = DEFAULT_BUCKET,
        docker_image:  str  = '',
        plan:          str  = DEFAULT_PLAN,
        region:        str  = DEFAULT_REGION,
        os_id:         int  = DEFAULT_OS_ID,
        dry_run:       bool = False,
    ) -> None:
        self.api_key       = vultr_api_key
        self.obj_endpoint  = obj_endpoint
        self.obj_access    = obj_access
        self.obj_secret    = obj_secret
        self.bucket        = bucket
        self.docker_image  = docker_image
        self.plan          = plan
        self.region        = region
        self.os_id         = os_id
        self.dry_run       = dry_run

        self.log = LoggingService('VultrTrainer')

        self._s3 = boto3.client(
            's3',
            endpoint_url         = obj_endpoint,
            aws_access_key_id    = obj_access,
            aws_secret_access_key= obj_secret,
            config               = Config(
                signature_version = 's3v4',
                retries           = {'max_attempts': 5, 'mode': 'standard'},
            ),
        )

        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type':  'application/json',
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def launch(
        self,
        manifest_path: Path,
        crops_dir:     Path,
        extra_train_args: str = '',
    ) -> str:
        """
        Upload data, provision an instance, and start training.

        :param manifest_path:    Local path to ``manifest.csv``.
        :param crops_dir:        Local root directory of PNG crops.
        :param extra_train_args: Extra arguments forwarded to
                                 ``train_cnn.py`` via
                                 ``EXTRA_TRAIN_ARGS`` env var.
        :return:                 Vultr instance ID (UUID string).
        """
        # 1 ── upload training data
        self._upload_manifest(manifest_path)
        self._upload_crops(crops_dir)

        # 2 ── ensure bucket exists
        self._ensure_bucket()

        # 3 ── build cloud-init user-data script
        user_data = self._build_user_data(extra_train_args)

        # 4 ── create instance
        instance_id = self._create_instance(user_data)
        self.log.info(f'Instance created: {instance_id}')

        # 5 ── wait until active
        self._wait_for_active(instance_id)
        self.log.info(f'Instance is active.  Training will start shortly.')
        self.log.info(
            f'Monitor via:  vultr-cli instance get {instance_id}\n'
            f'Or SSH in and: journalctl -u docker -f'
        )

        return instance_id

    def destroy(self, instance_id: str) -> None:
        """
        Destroy a Vultr instance.

        :param instance_id: Vultr instance UUID.
        :return:            None.
        """
        url = f'{VULTR_API_BASE}/instances/{instance_id}'
        self.log.info(f'Destroying instance {instance_id} ...')

        if self.dry_run:
            self.log.info('[dry-run] Would DELETE ' + url)
            return

        resp = self._session.delete(url)
        if resp.status_code not in (200, 204):
            self.log.warn(
                f'Destroy returned HTTP {resp.status_code}: {resp.text}'
            )
        else:
            self.log.info(f'Instance {instance_id} destroyed.')

    def upload_crops_only(self, crops_dir: Path) -> None:
        """
        Upload PNG crops without launching an instance.  Useful for a
        one-time pre-upload before the first training run.

        :param crops_dir: Local root directory of PNG crops.
        :return:          None.
        """
        self._ensure_bucket()
        self._upload_crops(crops_dir)

    # ------------------------------------------------------------------
    # Private helpers — data upload
    # ------------------------------------------------------------------

    def _ensure_bucket(self) -> None:
        """Create the S3 bucket if it does not already exist."""
        try:
            self._s3.head_bucket(Bucket=self.bucket)
            self.log.info(f'Bucket already exists: {self.bucket}')
        except self._s3.exceptions.ClientError:
            self.log.info(f'Creating bucket: {self.bucket}')
            if not self.dry_run:
                self._s3.create_bucket(Bucket=self.bucket)

    def _upload_manifest(self, manifest_path: Path) -> None:
        """
        Upload the manifest CSV to object storage.

        :param manifest_path: Local path to ``manifest.csv``.
        """
        self.log.info(
            f'Uploading manifest: {manifest_path} → '
            f's3://{self.bucket}/{MANIFEST_KEY}'
        )
        if self.dry_run:
            return
        self._s3.upload_file(
            str(manifest_path),
            self.bucket,
            MANIFEST_KEY,
            ExtraArgs={'ContentType': 'text/csv'},
        )

    def _upload_crops(self, crops_dir: Path) -> None:
        """
        Upload the PNG spectrogram crops to object storage using parallel
        multipart transfers.  Skips objects that already exist with the
        same file size (cheap resume logic).

        :param crops_dir: Local root directory containing date-partition
                          subdirectories of PNG files.
        """
        self.log.info(f'Inventorying existing objects in s3://{self.bucket}/{CROPS_PREFIX} ...')
        existing: dict[str, int] = {}
        paginator = self._s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=CROPS_PREFIX):
            for obj in page.get('Contents', []):
                existing[obj['Key']] = obj['Size']

        self.log.info(f'  {len(existing):,} objects already in bucket.')

        png_files = list(crops_dir.rglob('*.png'))
        self.log.info(f'  {len(png_files):,} PNG files to sync.')

        skipped = uploaded = 0
        for png in png_files:
            rel      = png.relative_to(crops_dir)
            obj_key  = CROPS_PREFIX + rel.as_posix()
            local_sz = png.stat().st_size

            if existing.get(obj_key) == local_sz:
                skipped += 1
                continue

            if not self.dry_run:
                self._s3.upload_file(str(png), self.bucket, obj_key)
            uploaded += 1

            if uploaded % 50_000 == 0:
                self.log.info(
                    f'  Upload progress: {uploaded:,} uploaded, '
                    f'{skipped:,} skipped ...'
                )

        self.log.info(
            f'Crop upload complete: {uploaded:,} uploaded, {skipped:,} skipped.'
        )

    # ------------------------------------------------------------------
    # Private helpers — instance management
    # ------------------------------------------------------------------

    def _build_user_data(self, extra_train_args: str = '') -> str:
        """
        Build the cloud-init user-data script that runs on first boot.

        The script:
        1. Installs Docker + NVIDIA Container Toolkit
        2. Pulls the training image
        3. Runs the container with the correct env vars
        4. The container's entrypoint handles data copy + torchrun +
           self-destruct.

        :param extra_train_args: Forwarded to ``EXTRA_TRAIN_ARGS``.
        :return:                 Cloud-init script as a string.
        """
        return f"""#!/usr/bin/env bash
set -euo pipefail

# ── Docker ───────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq docker.io

# ── NVIDIA Container Toolkit ─────────────────────────────────────────
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ── pull and run training container ──────────────────────────────────
docker pull {self.docker_image}

docker run --rm --gpus all \\
    -e VULTR_API_KEY="{self.api_key}" \\
    -e OBJECT_STORE_BUCKET="{self.bucket}" \\
    -e OBJECT_STORE_ENDPOINT="{self.obj_endpoint}" \\
    -e OBJECT_STORE_ACCESS="{self.obj_access}" \\
    -e OBJECT_STORE_SECRET="{self.obj_secret}" \\
    -e MANIFEST_KEY="{MANIFEST_KEY}" \\
    -e MODEL_OUT_KEY_PREFIX="{MODEL_OUT_PREFIX}" \\
    -e VULTR_INSTANCE_ID="$(curl -s http://169.254.169.254/v1.json | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"instance-v2-uuid\"])')" \\
    -e EXTRA_TRAIN_ARGS="{extra_train_args}" \\
    -v /nvme:/data \\
    {self.docker_image}
"""

    def _create_instance(self, user_data: str) -> str:
        """
        Create a Vultr instance and return its ID.

        :param user_data: Cloud-init script for first-boot.
        :return:          Instance UUID.
        """
        payload = {
            'region':    self.region,
            'plan':      self.plan,
            'os_id':     self.os_id,
            'label':     'bats-cnn-training',
            'user_data': user_data,
            'tags':      ['bats', 'cnn', 'training'],
        }

        self.log.info(
            f'Creating instance: plan={self.plan}, region={self.region}'
        )

        if self.dry_run:
            self.log.info('[dry-run] Would POST to ' + VULTR_API_BASE + '/instances')
            return 'dry-run-instance-id'

        resp = self._session.post(
            f'{VULTR_API_BASE}/instances',
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        return resp.json()['instance']['id']

    def _wait_for_active(self, instance_id: str) -> None:
        """
        Poll until the instance status is ``active`` or the timeout is
        reached.

        :param instance_id: Vultr instance UUID.
        :raises RuntimeError: If the instance does not become active
                              within :data:`LAUNCH_TIMEOUT` seconds.
        """
        url      = f'{VULTR_API_BASE}/instances/{instance_id}'
        deadline = time.monotonic() + LAUNCH_TIMEOUT
        self.log.info('Waiting for instance to become active ...')

        while time.monotonic() < deadline:
            resp   = self._session.get(url)
            resp.raise_for_status()
            status = resp.json()['instance']['status']
            power  = resp.json()['instance'].get('power_status', '')
            self.log.info(f'  status={status}  power={power}')

            if status == 'active' and power == 'running':
                return

            time.sleep(POLL_INTERVAL)

        raise RuntimeError(
            f'Instance {instance_id} did not become active within '
            f'{LAUNCH_TIMEOUT}s.'
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed namespace.
    """
    p = argparse.ArgumentParser(
        prog='vultr_trainer.py',
        description=(
            'Vultr GPU VM lifecycle manager for bat chirp CNN training.\n\n'
            'Modes:\n'
            '  --launch   Upload data + create instance + start training\n'
            '  --destroy  Destroy an existing instance\n'
            '  --upload   Upload crops/manifest only (no VM)\n'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── mode ──────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch',  action='store_true', help='Provision + train.')
    mode.add_argument('--destroy', action='store_true', help='Destroy instance.')
    mode.add_argument('--upload',  action='store_true', help='Upload data only.')

    # ── credentials ───────────────────────────────────────────────────
    p.add_argument(
        '--vultr-api-key',
        default=os.environ.get('VULTR_API_KEY', ''),
        help='Vultr API key (default: $VULTR_API_KEY).',
    )
    p.add_argument(
        '--obj-access',
        default=os.environ.get('OBJECT_STORE_ACCESS', ''),
        help='Object Storage access key (default: $OBJECT_STORE_ACCESS).',
    )
    p.add_argument(
        '--obj-secret',
        default=os.environ.get('OBJECT_STORE_SECRET', ''),
        help='Object Storage secret key (default: $OBJECT_STORE_SECRET).',
    )
    p.add_argument(
        '--obj-endpoint',
        default=os.environ.get('OBJECT_STORE_ENDPOINT', 'https://sjc1.vultrobjects.com'),
        help='Object Storage endpoint URL.',
    )

    # ── data paths (launch / upload modes) ────────────────────────────
    p.add_argument(
        '--manifest',
        type=Path,
        default=Path('/qnap/bats/jr_pipeline/data/bat_crops/manifest.csv'),
        help='Path to manifest.csv.',
    )
    p.add_argument(
        '--crops-dir',
        type=Path,
        default=Path('/qnap/bats/jr_pipeline/data/bat_crops'),
        help='Root directory of PNG crops.',
    )
    p.add_argument(
        '--bucket',
        default=DEFAULT_BUCKET,
        help=f'Object Storage bucket name (default: {DEFAULT_BUCKET}).',
    )

    # ── instance options (launch mode) ────────────────────────────────
    p.add_argument(
        '--docker-image',
        default='',
        help='Docker image reference, e.g. registry.example.com/bats-train:latest.',
    )
    p.add_argument(
        '--plan',
        default=DEFAULT_PLAN,
        help=f'Vultr plan ID (default: {DEFAULT_PLAN}).',
    )
    p.add_argument(
        '--region',
        default=DEFAULT_REGION,
        help=f'Vultr region code (default: {DEFAULT_REGION}).',
    )
    p.add_argument(
        '--extra-train-args',
        default='',
        help='Extra arguments forwarded to train_cnn.py.',
    )

    # ── destroy mode ──────────────────────────────────────────────────
    p.add_argument(
        '--instance-id',
        default='',
        help='Instance UUID to destroy (--destroy mode).',
    )

    # ── misc ──────────────────────────────────────────────────────────
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='Log actions without making API calls or uploading data.',
    )

    return p.parse_args()


def main() -> None:
    """
    Entry point.

    :return: None.
    """
    args = _parse_args()

    trainer = VultrTrainer(
        vultr_api_key = args.vultr_api_key,
        obj_endpoint  = args.obj_endpoint,
        obj_access    = args.obj_access,
        obj_secret    = args.obj_secret,
        bucket        = args.bucket,
        docker_image  = args.docker_image,
        plan          = args.plan,
        region        = args.region,
        dry_run       = args.dry_run,
    )

    if args.launch:
        if not args.docker_image:
            print('--docker-image is required for --launch', file=sys.stderr)
            sys.exit(1)
        instance_id = trainer.launch(
            manifest_path    = args.manifest,
            crops_dir        = args.crops_dir,
            extra_train_args = args.extra_train_args,
        )
        print(f'Instance ID: {instance_id}')

    elif args.destroy:
        if not args.instance_id:
            print('--instance-id is required for --destroy', file=sys.stderr)
            sys.exit(1)
        trainer.destroy(args.instance_id)

    elif args.upload:
        trainer.upload_crops_only(args.crops_dir)
        trainer._upload_manifest(args.manifest)  # noqa: SLF001


if __name__ == '__main__':
    main()
