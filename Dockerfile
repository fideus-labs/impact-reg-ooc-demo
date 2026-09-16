# The environment the brainlife.io App runs in (main -> singularity exec ... brainlife/run.py).
#
# brainlife's guidance is to containerize the dependencies, not the App: this image holds the two pixi
# environments of pixi.toml (default: torch 2.12, FireANTs, ConvexAdam/ITKIMPACT; elastix: torch 2.8, the
# LibTorch the elastix-IMPACT binary links), the elastix-IMPACT binary and the feature model the presets
# fetch from Hugging Face, so a compute node without internet access still runs every preset. The App's
# own code (main, brainlife/run.py, pipeline/, presets/) stays in the git repository brainlife clones.
#
#   docker build -t ghcr.io/fideus-labs/impact-reg-ooc-demo:dev .
#   IMPACT_REG_OOC_IMAGE=docker-daemon://ghcr.io/fideus-labs/impact-reg-ooc-demo:dev ./main
#
# The torches are CUDA 12 wheels that carry their own runtime; the host contributes the driver
# (singularity --nv), so the base image needs no CUDA of its own.
FROM ghcr.io/prefix-dev/pixi:0.81.0-noble

ARG ELASTIX_IMPACT_RELEASE=1.0.0
# Every `VBoussot/impact-torchscript-models:<file>` presets/*/Prediction.yml references (grep it to refresh).
ARG IMPACT_FEATURE_MODELS="TS/M291.pt MIND/R1D2_2D.pt MIND/R1D2_3D.pt MIND/R2D2_2D.pt MIND/R2D2_3D.pt"

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/impact-reg-ooc-demo
COPY pixi.toml pixi.lock ./

# Both environments, exactly as locked. PIXI_HOME on a path outside $HOME: singularity mounts the host's
# $HOME over the image's, which would hide anything installed there.
ENV PIXI_HOME=/opt/pixi PIXI_CACHE_DIR=/opt/pixi/cache
RUN pixi install --locked -e default -e elastix && rm -rf /opt/pixi/cache

# Activation hooks: `bash /opt/impact-reg-ooc-demo/env/<env>.sh <command...>` runs a command in an
# environment, with pixi.toml's activation env (KONFAI_ELASTIX_EXTRA_LIB for elastix) applied.
RUN mkdir -p env \
    && for e in default elastix; do \
         pixi shell-hook -e "$e" -s bash > "env/$e.sh" && printf '\nexec "$@"\n' >> "env/$e.sh"; \
       done

# The elastix-IMPACT binary (CUDA 12.8 flavour, the LibTorch 2.8 of the elastix environment).
# impact-reg-konfai would download it into ~/.cache/konfai on first use; KONFAI_ELASTIX_DIR points at
# this copy instead.
RUN mkdir -p /opt/elastix-impact \
    && curl -fsSL -o /tmp/elastix.zip \
       "https://github.com/vboussot/ImpactElastix/releases/download/${ELASTIX_IMPACT_RELEASE}/elastix-impact-linux-x86_64-cu128.zip" \
    && unzip -q /tmp/elastix.zip -d /opt/elastix-impact && rm /tmp/elastix.zip \
    && chmod +x /opt/elastix-impact/bin/elastix /opt/elastix-impact/bin/transformix
ENV KONFAI_ELASTIX_DIR=/opt/elastix-impact

# The IMPACT feature model(s) the presets reference (presets/*/Prediction.yml, `ref:
# VBoussot/impact-torchscript-models:<file>`), into a cache brainlife/run.py seeds the user's from.
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN MODELS="${IMPACT_FEATURE_MODELS}" HF_HOME=/opt/impact-reg-ooc-demo/hf-cache bash env/default.sh python -c "\
import os; from huggingface_hub import hf_hub_download; \
[hf_hub_download('VBoussot/impact-torchscript-models', f) for f in os.environ['MODELS'].split()]" \
    && rm -rf /opt/impact-reg-ooc-demo/hf-cache/hub/.locks

# The presets' requirements.txt (hydra-core, nibabel, pandas) are already in the lock; tell konfai-apps
# not to pip-install anything at run time (the image is read-only under singularity).
ENV KONFAI_APPS_INSTALL_REQUIREMENTS=0

# brainlife's last step for any App image: refresh the loader cache.
RUN ldconfig

LABEL org.opencontainers.image.source="https://github.com/fideus-labs/impact-reg-ooc-demo" \
      org.opencontainers.image.description="Out-of-core multimodal registration with KonfAI and IMPACT: the environment of the brainlife.io App"
