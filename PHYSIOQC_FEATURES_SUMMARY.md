# PhysioQC - Complete Feature Summary

## 🎯 **Project Overview**
**PhysioQC** is a professional-grade, headless CLI tool for real-time Signal Quality Assessment (SQI) and auto-event logging on physiological signals. Built for researchers, clinicians, and engineers working with neural data.

**Tagline**: Stream in → SQIs & auto‑events out

---

## 🏗️ **Phase 1: Signal Quality First (SQI Core)**

### **Composite SQI Score (0-100)**
- **Weighted combination** of all quality metrics
- **Real-time scoring**: RMS, DC offset, line noise, HF/LF ratio, clipping, flatline, dropout
- **Instant insights**: One number per channel for quick assessment
- **Professional thresholds**: Configurable quality boundaries

### **Live Status Monitoring**
- **Real-time table**: Shows worst channels with color coding
- **Auto-refreshing**: Updates every 1-2 seconds
- **Professional display**: Rich terminal interface with tables
- **Configurable**: Adjustable refresh rates and channel counts

### **Core SQI Metrics**
- **RMS**: Root Mean Square amplitude
- **Line Noise Ratio**: Power at mains frequency (50/60Hz)
- **DC Offset**: Mean signal level
- **HF/LF Ratio**: High-frequency to low-frequency power ratio
- **Clipping Percentage**: Signal saturation detection
- **Flatline Ratio**: Signal variation assessment
- **Dropout Count**: Data loss detection

---

## 🧠 **Phase 2: Neural Data In → Decode Out**

### **External Decoder Integration**
- **Python modules**: Direct import and execution
- **Subprocess support**: Command-line decoder integration
- **Socket communication**: Network-based decoder support
- **Timeout handling**: Robust error management

### **Performance Metrics**
- **Accuracy tracking**: Real-time decoder performance
- **Latency measurement**: Processing time analysis
- **Throughput monitoring**: Hz-based performance metrics
- **Confidence scoring**: Decoder output reliability

### **Decoder Output Capture**
- **Structured data**: JSON format with timestamps
- **Prediction logging**: Classification results tracking
- **Metadata preservation**: Session and window information
- **Export ready**: Parquet format for analysis

---

## 🌐 **Phase 3: Multi-Dataset Ready**

### **Data Source Support**
- **File-based**: .npy, .csv with automatic format detection
- **LSL Streams**: Real-time LabStreamingLayer integration
- **Simulated data**: Parameterized synthetic signal generation
- **BCI Competition**: Dataset loading framework

### **Source Management**
- **Stream discovery**: Automatic LSL stream detection
- **Parameter parsing**: Flexible configuration options
- **Error handling**: Graceful fallback mechanisms
- **Documentation**: Built-in help and examples

### **Supported Formats**
- **NumPy arrays**: .npy files with metadata
- **CSV data**: Comma-separated values
- **LSL streams**: OpenBCI, g.tec, and other devices
- **Simulated**: Customizable synthetic signals

---

## 🏗️ **Phase 4: Engineering Excellence**

### **Export Capabilities**
- **NumPy format**: .npz files for Python analysis
- **MATLAB format**: .mat files for MATLAB integration
- **Structured data**: Channel-wise organization
- **Metadata preservation**: Sampling rates and timestamps

### **Benchmark Mode**
- **Comprehensive testing**: SQI and decoder performance
- **Statistical analysis**: Mean, std, min, max metrics
- **JSON output**: Machine-readable results
- **Comparison ready**: Pre/post analysis support

### **Low-Friction Deployment**
- **Dry-run mode**: Connectivity testing without saving
- **Instant validation**: Quick source verification
- **Error reporting**: Clear failure messages
- **Command suggestions**: Ready-to-use examples

---

## 🚀 **Phase 5: Advanced Visualization & Testing**

### **Live Web Dashboard**
- **FastAPI backend**: Modern web framework
- **WebSocket streaming**: Real-time data updates
- **Plotly.js visualization**: Interactive charts
- **Dark theme**: Professional UI design

### **Dashboard Features**
- **SQI Heatmap**: Channel quality visualization
- **Decoder plots**: Confidence over time
- **Live metrics**: Current performance indicators
- **Responsive design**: Works on any screen size

### **Interactive Annotation**
- **Hotkey system**: Keyboard-driven annotation
- **Customizable mappings**: User-defined key bindings
- **Real-time feedback**: Instant visual confirmation
- **Event logging**: Structured annotation storage

### **Signal Validation**
- **Noise injection**: Controlled fault testing
- **Pattern application**: Line burst, dropout, clipping
- **Impact analysis**: Quality degradation measurement
- **Channel targeting**: Selective fault application

---

## 🛠️ **Technical Architecture**

### **Core Dependencies**
- **Typer**: Modern CLI framework
- **NumPy/SciPy**: Numerical computing
- **Pandas**: Data manipulation
- **PyArrow**: Fast data storage
- **Rich**: Terminal formatting
- **FastAPI**: Web dashboard
- **PyLSL**: Stream integration

### **Data Flow**
1. **Source Loading**: Multi-format data ingestion
2. **Window Processing**: Sliding window analysis
3. **SQI Computation**: Real-time quality metrics
4. **Event Detection**: Automated artifact identification
5. **Output Generation**: Structured data export

### **File Structure**
```
runs/session_name/
├── sqi.parquet          # Signal quality metrics
├── events.jsonl         # Automated and manual events
├── decode.parquet       # Decoder outputs (if applicable)
├── decode_metrics.json  # Performance metrics
├── provenance.json      # Processing metadata
├── export.npz          # NumPy export
└── export.mat          # MATLAB export
```

---

## 🎯 **Use Cases**

### **Research Applications**
- **BCI Research**: Real-time quality monitoring
- **Clinical Studies**: Signal validation
- **Algorithm Development**: Performance benchmarking
- **Data Collection**: Quality assurance

### **Clinical Applications**
- **EEG Monitoring**: Real-time quality assessment
- **Neuromodulation**: Signal validation
- **Diagnostic Tools**: Quality metrics
- **Research Protocols**: Standardized assessment

### **Engineering Applications**
- **Hardware Testing**: Device validation
- **Algorithm Validation**: Performance testing
- **System Integration**: Quality monitoring
- **Development Workflows**: Continuous testing

---

## 🚀 **Getting Started**

### **Quick Start**
```bash
# Install
pip install physioqc

# Generate demo data
physioq simulate --out demo.npy

# Run analysis
physioq ingest --source "file:demo.npy" --out-dir runs/demo

# View results
physioq status runs/demo
```

### **Advanced Usage**
```bash
# LSL stream processing
physioq ingest --source "lsl:OpenBCI" --duration 30

# With decoder integration
physioq ingest --source "file:data.npy" --decode "python:my_decoder"

# Live monitoring
physioq dashboard --session runs/my_session

# Export for analysis
physioq to-npz runs/my_session
physioq to-mat runs/my_session
```

---

## 📊 **Performance Characteristics**

### **Processing Speed**
- **Real-time capable**: Sub-second processing
- **Scalable**: Handles 100+ channels
- **Efficient**: Minimal memory footprint
- **Robust**: Error handling and recovery

### **Accuracy**
- **Validated metrics**: Peer-reviewed SQI algorithms
- **Configurable thresholds**: Adaptable to use cases
- **Statistical rigor**: Proper normalization and scaling
- **Reproducible**: Deterministic processing

### **Reliability**
- **Error handling**: Graceful failure modes
- **Data integrity**: Checksums and validation
- **Provenance tracking**: Complete audit trail
- **Backup mechanisms**: Multiple output formats

---

## 🔮 **Future Roadmap**

### **Planned Features**
- **Machine Learning**: Automated artifact detection
- **Cloud Integration**: Remote processing capabilities
- **Real-time Streaming**: Live data processing
- **Advanced Visualization**: 3D brain mapping

### **Integration Targets**
- **BCI2000**: Protocol integration
- **OpenVibe**: Workflow compatibility
- **FieldTrip**: MATLAB toolbox integration
- **MNE-Python**: Python ecosystem integration

---

## 📈 **Impact & Benefits**

### **Research Efficiency**
- **Time savings**: Automated quality assessment
- **Standardization**: Consistent metrics across studies
- **Reproducibility**: Complete processing records
- **Collaboration**: Shared analysis protocols

### **Clinical Applications**
- **Quality assurance**: Real-time monitoring
- **Standardization**: Consistent assessment
- **Documentation**: Complete audit trails
- **Integration**: Existing workflow compatibility

### **Engineering Benefits**
- **Development speed**: Rapid prototyping
- **Testing efficiency**: Automated validation
- **Quality control**: Continuous monitoring
- **Documentation**: Complete processing records

---

*PhysioQC represents a comprehensive solution for physiological signal quality assessment, combining real-time processing, professional visualization, and robust engineering practices to deliver a tool that meets the needs of researchers, clinicians, and engineers working with neural data.*
