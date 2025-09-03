# Vebo - Data Profiling & Quality Analysis

Vebo is a comprehensive data profiling system that automatically generates Python code from rules and executes quality checks against datasets.

## 🏗️ Project Structure

```
vebo/
├── python/                    # Python profiler implementation
│   ├── vebo_profiler/        # Core profiler package
│   └── requirements.txt      # Python dependencies
├── website/                  # Web interface (React/TypeScript)
├── rules/                    # Data quality rules and configurations
│   ├── data-quality/        # Data quality specific rules
│   ├── data-types/          # Data type validation rules
│   ├── cross-column/        # Cross-column relationship rules
│   └── table-level/         # Table-level analysis rules
├── examples/                 # Usage examples and sample data
│   ├── python/              # Python examples
│   ├── jupyter/             # Jupyter notebook examples
│   └── datasets/            # Sample datasets
├── docs/                     # Documentation
│   ├── api/                 # API documentation
│   ├── guides/              # User guides
│   └── examples/            # Example documentation
├── tests/                    # Test suites
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test data
└── cursor-rules-directory/   # Cursor AI rules collection
```

## 🚀 Quick Start

### Python Profiler

1. **Install dependencies:**
   ```bash
   cd python
   pip install -r requirements.txt
   ```

2. **Run the example:**
   ```bash
   cd examples/python
   python example_usage.py
   ```

### Web Interface

1. **Install dependencies:**
   ```bash
   cd website
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

## 📊 Features

- **Automatic Data Profiling:** Comprehensive analysis of data quality, types, and patterns
- **Rule-Based Engine:** Configurable rules for different data quality checks
- **Cross-Column Analysis:** Relationship detection between columns
- **Web Interface:** User-friendly dashboard for data exploration
- **Extensible Architecture:** Easy to add new rules and checks

## 📚 Documentation

- [API Documentation](docs/api/)
- [User Guides](docs/guides/)
- [Examples](docs/examples/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.