# PhyFPS-Bench-Gen

**PhyFPS-Bench-Gen** is a benchmark for auditing the temporal consistency of video generative models. It measures the alignment between a model's nominal frame rate (Meta FPS) and the physical motion speed (PhyFPS) predicted by our [Visual Chronometer](../README.md).

## Prompts

`prompts.txt` contains 100 text-to-video prompts designed for robust PhyFPS evaluation. Prompts are balanced across:

- **Primary entity**: human, animal, vehicle, nature
- **Motion type**: articulated, rigid-body, fluid, multi-agent
- **Camera behavior**: static, pan, tracking
- **Environmental effects**: rain, fire, wind
- **Scene context**: indoor, urban, nature

All prompts require at least one clearly dynamic instance and **avoid** speed-manipulation keywords (*slow motion*, *time-lapse*, *speed up*, etc.).

## Usage

### 1. Generate videos from prompts

Use your video generation model to produce one video per prompt under **default settings**:

```bash
# Example with your model
python your_model_inference.py --prompts prompts.txt --output_dir generated_videos/
```

### 2. Predict PhyFPS with Visual Chronometer

```bash
cd ../inference
python predict.py \
    --video_dir ../PhyFPS-Bench-Gen/generated_videos/ \
    --stride 4 \
    --output_csv ../PhyFPS-Bench-Gen/results.csv
```

### 3. Compute metrics

For each generated video, record the model's **Meta FPS** ($F_{\text{meta}}$) from official documentation or output metadata. Then compute:

| Metric | Formula | Measures |
|--------|---------|----------|
| **Avg. Error (FPS)** | $\frac{1}{V}\sum\|\bar{f}_v - F_{\text{meta}}\|$ | Meta-vs-PhyFPS alignment |
| **Pct. Error (%)** | $\frac{100}{V}\sum\frac{\|\bar{f}_v - F_{\text{meta}}\|}{F_{\text{meta}}}$ | Relative alignment |
| **Inter-video CV** | $\frac{\text{Std}(\bar{f}_v)}{\text{Mean}(\bar{f}_v)}$ | Cross-video consistency |
| **Intra-video CV** | $\frac{1}{V}\sum\frac{\text{Std}(\hat{f}_{v,c})}{\text{Mean}(\hat{f}_{v,c})}$ | Within-video stability |

Where $\bar{f}_v$ is the average PhyFPS across all sliding-window clips for video $v$, and $\hat{f}_{v,c}$ is the per-clip prediction.
