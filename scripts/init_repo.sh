# scripts/init_repo.sh -- run once from repo root
#!/usr/bin/env bash
set -e

# Core source directories
mkdir -p data/echonet/{raw,processed,splits}
mkdir -p data/camus/{raw,processed,splits}
mkdir -p mvm/{masking,encoder,decoder,utils}
mkdir -p validation/{scripts,results,plots}
mkdir -p interpretability/{shap,attention,gradcam,uncertainty,plots}
mkdir -p secure_ai/{privacy,encryption,compliance,distillation,api}
mkdir -p vr_module/{web,unity_skeleton,export,api}
mkdir -p experiments/{configs,logs,checkpoints}
mkdir -p notebooks docker tests/{unit,integration} docs .github/{workflows,ISSUE_TEMPLATE}

# __init__.py files
touch mvm/__init__.py mvm/masking/__init__.py mvm/encoder/__init__.py
touch mvm/decoder/__init__.py mvm/utils/__init__.py
touch validation/__init__.py interpretability/__init__.py secure_ai/__init__.py

echo 'Directory structure created'
find . -type d | sort

