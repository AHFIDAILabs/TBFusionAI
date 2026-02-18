# TBFusionAI Setup Guide

## 🚀 Quick Start

The system requires trained ML models before it can make predictions. Follow these steps:

### Option 1: Docker (Recommended)

```bash
# 1. Build the Docker image
docker-compose -f docker/docker-compose.yml build

# 2. Start the container
docker-compose -f docker/docker-compose.yml up -d

# 3. Run the complete ML pipeline (takes ~2 hours)
docker exec tbfusionai-api python main.py run-pipeline

# 4. Check status
docker exec tbfusionai-api python main.py status

# 5. Access the API
# - Web UI: http://localhost:8000
# - API Docs: http://localhost:8000/api/docs
# - Health: http://localhost:8000/api/v1/health
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline
python main.py run-pipeline

# 4. Start the API server
python main.py serve

# 5. Access at http://localhost:8000
```

## 📋 Pipeline Stages

The ML pipeline consists of 4 stages:

| Stage | Description | Duration | Output |
|-------|-------------|----------|--------|
| **1. Data Ingestion** | Download CODA TB dataset from Hugging Face | ~15 min | Raw audio files & metadata |
| **2. Data Processing** | Extract Wav2Vec2 features, preprocess audio, balance classes | ~60 min | Feature embeddings CSV |
| **3. Model Training** | Train multiple ML models (CatBoost, XGBoost, LightGBM, etc.) | ~30 min | Trained model files |
| **4. Model Evaluation** | Create ensemble model with cost-sensitive optimization | ~5 min | Final ensemble model |

**Total time: ~2 hours**

## 🔍 Checking Status

### Using CLI

```bash
# Check which stages are complete
python main.py status

# View system information
python main.py info
```

### Using API

```bash
# Health check (works without models)
curl http://localhost:8000/api/v1/health

# Detailed status with setup instructions
curl http://localhost:8000/api/v1/status
```

## 🎯 Running Individual Stages

You can run stages individually if needed:

```bash
# Stage 1: Download dataset
python main.py ingest-data

# Stage 2: Process audio and extract features
python main.py process-data

# Stage 3: Train ML models
python main.py train-models

# Stage 4: Create ensemble model
python main.py evaluate-models

# Or run all stages at once
python main.py run-pipeline
```

### Smart Skip Logic

The pipeline automatically skips completed stages:

```bash
# First run: Executes all 4 stages
python main.py run-pipeline

# Second run: Skips all (already complete)
python main.py run-pipeline

# Force re-run everything
python main.py run-pipeline --force

# Force re-run specific stage
python main.py train-models --force
```

## 🐳 Docker Commands

```bash
# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop container
docker-compose -f docker/docker-compose.yml down

# Rebuild image (after code changes)
docker-compose -f docker/docker-compose.yml build --no-cache

# Execute commands inside container
docker exec tbfusionai-api python main.py status
docker exec tbfusionai-api python main.py run-pipeline

# Access container shell
docker exec -it tbfusionai-api bash
```

## 📊 Expected Output

After successful pipeline completion:

```
✓ All pipeline stages complete! Ready for predictions.

Pipeline Status
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Stage                  ┃   Status    ┃ Description                        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1. Data Ingestion      │ ✅ Complete │ Download CODA TB dataset           │
│ 2. Data Processing     │ ✅ Complete │ Extract audio features & balance   │
│ 3. Model Training      │ ✅ Complete │ Train multiple ML models           │
│ 4. Model Evaluation    │ ✅ Complete │ Create ensemble model              │
└────────────────────────┴─────────────┴────────────────────────────────────┘
```

## 🔧 Troubleshooting

### Models Not Loading

**Error:** `No such file or directory: 'artifacts/trained_models/cost_sensitive_ensemble_model.joblib'`

**Solution:** Run the pipeline to train models:
```bash
docker exec tbfusionai-api python main.py run-pipeline
```

### Pipeline Stuck on Data Processing

**Issue:** Audio feature extraction takes a long time

**Normal:** This stage processes hundreds of audio files and can take 60+ minutes. Progress is logged.

### Out of Memory

**Issue:** Pipeline crashes during training

**Solution:** 
- Increase Docker memory limit (8GB+ recommended)
- Or run locally with more RAM
- Or reduce batch size in `src/config.py`

### Download Fails

**Issue:** Cannot download dataset from Hugging Face

**Solution:**
- Check internet connection
- Verify Hugging Face is accessible
- Try again (pipeline has retry logic)

## 📁 Directory Structure

```
TBFusionAI/
├── artifacts/
│   ├── dataset/                 # Downloaded dataset
│   │   ├── raw_data/           # Audio files
│   │   └── meta_data/          # Metadata CSVs
│   ├── preprocessed_data/      # Processed audio
│   ├── labeled_data/           # Features + labels
│   ├── trained_models/         # Model files
│   │   ├── cost_sensitive_ensemble_model.joblib  # MAIN MODEL
│   │   ├── scaler.joblib
│   │   └── training_metadata.joblib
│   ├── reports/                # Evaluation reports
│   └── logs/                   # Application logs
├── src/                        # Source code
├── frontend/                   # Web UI
└── docker/                     # Docker configs
```

## 🎓 Next Steps

After pipeline completion:

1. **Test the API:**
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. **Access Web UI:**
   - Navigate to http://localhost:8000
   - Try the prediction page with sample audio

3. **Read API Documentation:**
   - Visit http://localhost:8000/api/docs
   - Explore available endpoints

4. **Make Predictions:**
   - Use the web interface
   - Or call API directly
   - Or use CLI: `python main.py predict <audio_file> --age 45 --sex Male ...`

## 💡 Tips

- **First-time setup:** Allocate 2-3 hours for complete pipeline
- **Subsequent runs:** Models persist, no need to retrain
- **Development:** Use `--reload` flag with `serve` command
- **Production:** Use Docker with proper resource limits
- **Monitoring:** Check logs in `artifacts/logs/tbfusionai.log`

## 📞 Support

If you encounter issues:

1. Check `artifacts/logs/tbfusionai.log` for detailed errors
2. Run `python main.py status` to see what's complete
3. Try `--force` flag to re-run failed stages
4. Open an issue on GitHub with logs and error messages
