# Vebo - Data Profiling & Quality Analysis

Vebo is a comprehensive data profiling system that automatically generates Python code from rules and executes quality checks against datasets.

## 🏗️ Project Structure

```
vebo/
├── python/                    # Python profiler implementation
│   ├── vebo_profiler/        # Core profiler package
│   │   └── core/            # Core modules (profiler, rule_engine, etc.)
│   ├── requirements.txt      # Python dependencies
│   ├── setup.py             # Package setup configuration  
│   └── server/              # FastAPI server implementation
├── website/                  # Web interface (React/TypeScript)
│   ├── src/                 # React source code
│   ├── components/          # UI components
│   └── dist/               # Build output
├── scripts/                  # Utility scripts
│   └── run_tests.py         # Test runner script
├── rules/                    # Data quality rules and configurations  
│   ├── data-quality/        # Data quality specific rules
│   ├── data-types/          # Data type validation rules
│   ├── cross-column/        # Cross-column relationship rules
│   └── table-level/         # Table-level analysis rules
├── examples/                 # Usage examples and sample data
│   ├── python/              # Python examples  
│   ├── jupyter/             # Jupyter notebook examples
│   └── datasets/            # Sample datasets and results
├── docs/                     # Documentation
│   ├── api/                 # API documentation
│   ├── guides/              # User guides
│   └── examples/            # Example documentation
├── tests/                    # Test suites
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── performance/         # Performance tests
│   ├── e2e/                 # End-to-end tests
│   └── fixtures/            # Test data and utilities
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

3. **Run the tests:**
   ```bash
   python scripts/run_tests.py
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

## 🧪 Testing

The project includes comprehensive test coverage:

- **Unit Tests:** Core component testing (`tests/unit/`)
- **Integration Tests:** Full workflow testing (`tests/integration/`)  
- **Performance Tests:** Optimization validation (`tests/performance/`)
- **End-to-End Tests:** Complete system testing (`tests/e2e/`)

Run tests with:
```bash
# All tests
python scripts/run_tests.py

# Specific test types  
python scripts/run_tests.py --type unit
python scripts/run_tests.py --type integration
python scripts/run_tests.py --type performance
```

## 📚 Documentation

- [API Documentation](docs/api/)
- [User Guides](docs/guides/)
- [Examples](docs/examples/)
- [Performance Optimizations](docs/PERFORMANCE_OPTIMIZATIONS_SUMMARY.md)
- [Enhancement Summary](docs/FOCUSED_ENHANCEMENTS_SUMMARY.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch  
3. Make your changes
4. Add comprehensive tests
5. Ensure all tests pass: `python scripts/run_tests.py`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.