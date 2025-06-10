# 🎨 MineGen Dashboard Preview

## Real-time Training Monitor
```
🎯 Training Monitor
┌─────────────────────────────────────────────────────────────┐
│ Training Status: 🟢 Training                               │
│ Progress: ████████████████████████████████████████ 85%     │
│ Epoch: 85/100 | Loss: 0.234 | Val Loss: 0.267            │
└─────────────────────────────────────────────────────────────┘

📈 Loss Curves                    ⚙️ Model Config
┌─────────────────────────────┐   ┌─────────────────────────┐
│     Loss vs Epoch           │   │ Model: VAE              │
│ 2.0 ┌─────────────────────┐ │   │ Latent Dim: 64          │
│     │ ╲                   │ │   │ Batch Size: 32          │
│ 1.5 │  ╲ Train Loss       │ │   │ Learning Rate: 1e-3     │
│     │   ╲                 │ │   │ Optimizer: Adam         │
│ 1.0 │    ╲___             │ │   │ Device: CUDA:0          │
│     │        ╲___         │ │   └─────────────────────────┘
│ 0.5 │            ╲___     │ │
│     │ Val Loss       ╲___ │ │   📊 Current Metrics
│ 0.0 └─────────────────────┘ │   ┌─────────────────────────┐
│     0   20   40   60   80   │   │ Reconstruction: 0.234   │
└─────────────────────────────┘   │ KL Divergence: 0.045    │
                                  │ Category Acc: 87.3%     │
                                  │ Samples/sec: 156        │
                                  └─────────────────────────┘
```

## Interactive Generation Viewer
```
🎨 Generation Viewer
┌─────────────────────────────────────────────────────────────┐
│ Generate New Samples                                        │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│ │ Samples: 10 │ │ Category: ▼ │ │ [Generate]  │           │
│ └─────────────┘ └─────────────┘ └─────────────┘           │
│                   Houses                                    │
│                   Castles                                   │
│                   Gardens                                   │
│                   Random                                    │
└─────────────────────────────────────────────────────────────┘

🏠 Generated Samples Gallery
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ [IMG]   │ │ [IMG]   │ │ [IMG]   │ │ [IMG]   │ │ [IMG]   │
│ House   │ │ Castle  │ │ Garden  │ │ Tower   │ │ Bridge  │
│ 16x16x16│ │ 24x32x24│ │ 20x8x20 │ │ 8x64x8  │ │ 32x8x16 │
│[Download│ │[Download│ │[Download│ │[Download│ │[Download│
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘

📈 Generation Queue: 3 pending | Last generated: 2 minutes ago
```

## Data Explorer
```
📊 Data Explorer
┌─────────────────────────────────────────────────────────────┐
│ Dataset Statistics                                          │
│ Total Schematics: 15,432 | Categories: 19 | Avg Size: 18x14x16 │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│ │[Download New│ │[Refresh Data│ │[Export Meta]│           │
│ └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘

📊 Size Distribution              📊 Category Distribution
┌─────────────────────────────┐   ┌─────────────────────────┐
│ Count                       │   │ Houses ████████████ 3.2k│
│   ▲                         │   │ Castles ████████ 2.1k   │
│ 3k│ ██                      │   │ Gardens ██████ 1.8k     │
│   │ ██                      │   │ Towers ████ 1.2k        │
│ 2k│ ██ ██                   │   │ Ships ███ 0.9k          │
│   │ ██ ██ ██                │   │ Others ████████ 2.3k    │
│ 1k│ ██ ██ ██ ██             │   └─────────────────────────┘
│   │ ██ ██ ██ ██ ██          │
│ 0 └─────────────────────────│   🔍 Sample Browser
│     8  16  32  64 128       │   ┌─────────────────────────┐
│     Size (blocks)           │   │ ID: 1337 | Castle      │
│                             │   │ Size: 48x64x32          │
└─────────────────────────────┘   │ Rating: ⭐⭐⭐⭐⭐        │
                                  │ Downloads: 2,341        │
                                  │ [Preview] [Download]    │
                                  └─────────────────────────┘
```

## System Status
```
⚡ System Status
┌─────────────────────────────────────────────────────────────┐
│ 🟢 Training Active | 🟡 Generation Queue: 3 | 🟢 Data: OK  │
│                                                             │
│ GPU: RTX 3080 (85% util) | Memory: 12.4/16GB | Temp: 72°C  │
│ CPU: 16 cores (45% util) | RAM: 24.1/32GB | Disk: 1.2TB   │
│                                                             │
│ Models: 4 checkpoints | Samples: 1,247 generated today     │
└─────────────────────────────────────────────────────────────┘

📈 Recent Activity
• 14:32 - Generated 10 castle samples
• 14:28 - Training epoch 85 completed (val_loss: 0.267)
• 14:25 - Downloaded 50 new schematics
• 14:20 - Started VAE training session
• 14:15 - Dashboard launched
```

## Features Highlight

### 🎯 Real-time Updates
- Live training metrics and loss curves
-Automatic refresh every 5 seconds
- WebSocket connections for instant updates
- Progress bars and status indicators

### 🎨 Interactive Generation
- Drag-and-drop parameter adjustment
- Category-specific generation
- Batch processing with queue management
- Preview and download generated schematics

### 📊 Data Insights
- Interactive plots with Plotly
- Dataset quality analysis
- Size and category distributions
- Sample browser with filtering

### ⚙️ Model Management
- Checkpoint comparison
- Hyperparameter visualization
- Performance metrics tracking
- Model architecture diagrams

### 🔧 System Monitoring
- Resource usage tracking
- Error logging and alerts
- Background task management
- Configuration hot-reloading