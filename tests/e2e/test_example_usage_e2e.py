"""
End-to-end tests using the example_usage.py script.
"""

import pytest
import subprocess
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))


class TestExampleUsageE2E:
    """End-to-end tests for the example usage script."""
    
    def test_example_usage_script_exists(self):
        """Test that the example usage script exists and is executable."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        assert example_path.exists(), "example_usage.py script not found"
        assert example_path.is_file(), "example_usage.py is not a file"
    
    def test_example_usage_imports(self):
        """Test that the example usage script can be imported without errors."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Test import by executing the script directly
        result = subprocess.run([
            sys.executable, str(example_path)
        ], capture_output=True, text=True, timeout=30)
        
        # Should not have import errors (may have other errors but imports should work)
        if result.returncode != 0:
            # Check if it's an import error vs other errors
            stderr = result.stderr.lower()
            if "import" in stderr and ("error" in stderr or "failed" in stderr):
                assert False, f"Import failed: {result.stderr}"
            # Other errors (like missing data files) are acceptable for import test
    
    def test_example_usage_script_execution(self):
        """Test that the example usage script runs successfully."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Run the script
        result = subprocess.run([
            sys.executable, str(example_path)
        ], capture_output=True, text=True, timeout=60)
        
        # Should complete successfully
        assert result.returncode == 0, f"Script execution failed: {result.stderr}"
        
        # Should produce output
        assert len(result.stdout) > 0, "Script produced no output"
        
        # Should contain expected output patterns
        output = result.stdout
        assert "📊 Loading Titanic dataset" in output or "📊 Creating sample dataset" in output
        assert "✅ Profiling completed" in output or "🎉 Demo completed successfully" in output
        assert "Overall Score:" in output
        assert "Quality Grade:" in output
    
    def test_example_usage_generates_output_file(self):
        """Test that the example usage script generates an output file."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Run the script
                result = subprocess.run([
                    sys.executable, str(example_path)
                ], capture_output=True, text=True, timeout=60)
                
                # Should complete successfully
                assert result.returncode == 0, f"Script execution failed: {result.stderr}"
                
                # Check for output files
                output_files = list(Path(temp_dir).glob("*.json"))
                assert len(output_files) > 0, "No JSON output files generated"
                
                # Check that the output file is valid JSON
                for output_file in output_files:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    # Should have expected structure
                    assert "metadata" in data
                    assert "summary" in data
                    assert "column_analysis" in data
                    assert "cross_column_analysis" in data
                    assert "table_level_analysis" in data
                    assert "errors" in data
                    
                    # Should have meaningful data
                    assert data["metadata"]["dataset_info"]["rows"] > 0
                    assert data["metadata"]["dataset_info"]["columns"] > 0
                    assert "overall_score" in data["summary"]
                    assert "quality_grade" in data["summary"]
                    
            finally:
                os.chdir(original_cwd)
    
    def test_example_usage_with_different_configurations(self):
        """Test example usage with different profiling configurations."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Test configurations
        configs = [
            {"deepness_level": "basic"},
            {"deepness_level": "standard"},
            {"deepness_level": "deep"},
            {"enable_cross_column": False},
            {"max_workers": 1},
            {"sample_size": 50}
        ]
        
        for config in configs:
            # Create a modified version of the script with different config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                # Read the original script
                with open(example_path, 'r') as f:
                    script_content = f.read()
                
                # Modify the configuration and add Python path
                config_str = ", ".join([f"{k}={repr(v)}" for k, v in config.items()])
                modified_content = script_content.replace(
                    "ProfilingConfig()",
                    f"ProfilingConfig({config_str})"
                )
                
                # Add Python path to the script
                python_path = str(example_path.parent.parent.parent / "python")
                modified_content = f"import sys\nsys.path.insert(0, '{python_path}')\n" + modified_content
                
                temp_script.write(modified_content)
                temp_script.flush()
                
                try:
                    # Run the modified script
                    result = subprocess.run([
                        sys.executable, temp_script.name
                    ], capture_output=True, text=True, timeout=60)
                    
                    # Should complete successfully
                    assert result.returncode == 0, f"Script failed with config {config}: {result.stderr}"
                    
                    # Should produce output
                    assert len(result.stdout) > 0, f"No output with config {config}"
                    
                finally:
                    # Clean up
                    os.unlink(temp_script.name)
    
    def test_example_usage_error_handling(self):
        """Test that the example usage script handles errors gracefully."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Test with invalid configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            # Read the original script
            with open(example_path, 'r') as f:
                script_content = f.read()
            
            # Modify to use invalid configuration
            modified_content = script_content.replace(
                "ProfilingConfig()",
                "ProfilingConfig(deepness_level='invalid_level')"
            )
            
            temp_script.write(modified_content)
            temp_script.flush()
            
            try:
                # Run the modified script
                result = subprocess.run([
                    sys.executable, temp_script.name
                ], capture_output=True, text=True, timeout=60)
                
                # Should handle error gracefully (either succeed or fail with clear message)
                if result.returncode != 0:
                    assert "error" in result.stderr.lower() or "exception" in result.stderr.lower()
                
            finally:
                # Clean up
                os.unlink(temp_script.name)
    
    def test_example_usage_performance(self):
        """Test that the example usage script completes within reasonable time."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        import time
        start_time = time.time()
        
        # Run the script
        result = subprocess.run([
            sys.executable, str(example_path)
        ], capture_output=True, text=True, timeout=120)  # 2 minute timeout
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete successfully
        assert result.returncode == 0, f"Script execution failed: {result.stderr}"
        
        # Should complete within reasonable time (less than 2 minutes)
        assert execution_time < 120, f"Script took too long: {execution_time:.2f} seconds"
        
        # Should complete within reasonable time for a demo (less than 30 seconds)
        assert execution_time < 30, f"Script took too long for demo: {execution_time:.2f} seconds"
    
    def test_example_usage_output_quality(self):
        """Test that the example usage script produces high-quality output."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Run the script
                result = subprocess.run([
                    sys.executable, str(example_path)
                ], capture_output=True, text=True, timeout=60)
                
                # Should complete successfully
                assert result.returncode == 0, f"Script execution failed: {result.stderr}"
                
                # Check for output files
                output_files = list(Path(temp_dir).glob("*.json"))
                assert len(output_files) > 0, "No JSON output files generated"
                
                # Analyze the output quality
                for output_file in output_files:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check metadata quality
                    metadata = data["metadata"]
                    assert metadata["dataset_info"]["rows"] > 0
                    assert metadata["dataset_info"]["columns"] > 0
                    assert metadata["execution_info"]["duration_seconds"] > 0
                    assert metadata["execution_info"]["checks_executed"] > 0
                    
                    # Check summary quality
                    summary = data["summary"]
                    assert 0 <= summary["overall_score"] <= 100
                    assert summary["quality_grade"] in ["A", "B", "C", "D", "F"]
                    # These fields might be integers or lists depending on implementation
                    assert isinstance(summary["critical_issues"], (list, int))
                    assert isinstance(summary["warnings"], (list, int))
                    assert isinstance(summary["recommendations"], (list, int))
                    
                    # Check column analysis quality
                    column_analysis = data["column_analysis"]
                    assert len(column_analysis) > 0
                    
                    for col_name, col_data in column_analysis.items():
                        assert "data_type" in col_data
                        assert "null_count" in col_data
                        assert "unique_count" in col_data
                        assert "checks" in col_data
                        assert isinstance(col_data["checks"], list)
                    
                    # Check that errors are properly handled
                    errors = data["errors"]
                    assert isinstance(errors, list)
                    
            finally:
                os.chdir(original_cwd)
    
    def test_example_usage_with_mock_data(self):
        """Test example usage with mocked data to ensure it works with different datasets."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Create a test script that uses mock data
        python_path = str(example_path.parent.parent.parent / "python")
        test_script_content = '''
import sys
import os
sys.path.insert(0, '{}')

import pandas as pd
import numpy as np
from vebo_profiler import VeboProfiler
from vebo_profiler.core.profiler import ProfilingConfig

def create_test_dataset():
    """Create a simple test dataset."""
    np.random.seed(42)
    n_rows = 100
    
    data = {{
        'id': range(1, n_rows + 1),
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(50000, 15000, n_rows),
        'name': ['Person_' + str(i) for i in range(n_rows)],
        'department': np.random.choice(['IT', 'HR', 'Finance'], n_rows),
        'is_active': np.random.choice([True, False], n_rows)
    }}
    
    return pd.DataFrame(data)

def main():
    """Main function to test the profiler."""
    print("🧪 Testing Vebo Profiler with mock data...")
    
    # Create test dataset
    df = create_test_dataset()
    print("   Created dataset with {{}} rows and {{}} columns".format(len(df), len(df.columns)))
    
    # Configure profiler
    config = ProfilingConfig(
        enable_cross_column=True,
        deepness_level="standard",
        max_workers=2
    )
    
    # Create profiler and run analysis
    profiler = VeboProfiler(config)
    
    try:
        result = profiler.profile_dataframe(df, filename="test_dataset.csv")
        
        # Save results
        output_file = "test_profiling_results.json"
        profiler.save_result(result, output_file)
        
        print("✅ Profiling completed successfully!")
        print("   Results saved to: {{}}".format(output_file))
        print("   Overall Score: {{:.1f}}".format(result.summary['overall_score']))
        print("   Quality Grade: {{}}".format(result.summary['quality_grade']))
        print("   Columns analyzed: {{}}".format(len(result.column_analysis)))
        print("   Checks executed: {{}}".format(result.metadata['execution_info']['checks_executed']))
        
    except Exception as e:
        print("❌ Profiling failed: {{}}".format(e))
        raise

if __name__ == "__main__":
    main()
'''.format(python_path)
        
        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Write the test script
                test_script_path = Path(temp_dir) / "test_script.py"
                with open(test_script_path, 'w') as f:
                    f.write(test_script_content)
                
                # Run the test script
                result = subprocess.run([
                    sys.executable, str(test_script_path)
                ], capture_output=True, text=True, timeout=60)
                
                # Should complete successfully
                assert result.returncode == 0, f"Test script failed: {result.stderr}"
                
                # Should produce expected output
                output = result.stdout
                assert "🧪 Testing Vebo Profiler with mock data" in output
                assert "✅ Profiling completed successfully" in output
                assert "Overall Score:" in output
                assert "Quality Grade:" in output
                
                # Should generate output file
                output_files = list(Path(temp_dir).glob("*.json"))
                assert len(output_files) > 0, "No JSON output files generated"
                
            finally:
                os.chdir(original_cwd)
    
    def test_example_usage_dependencies(self):
        """Test that the example usage script has all required dependencies."""
        example_path = Path(__file__).parent.parent.parent / "examples" / "python" / "example_usage.py"
        
        # Read the script to check for imports
        with open(example_path, 'r') as f:
            script_content = f.read()
        
        # Check for required imports
        required_imports = [
            "pandas",
            "numpy", 
            "vebo_profiler",
            "ProfilingConfig"
        ]
        
        for import_name in required_imports:
            assert import_name in script_content, f"Required import '{import_name}' not found in script"
        
        # Test that the script can be imported without dependency errors
        result = subprocess.run([
            sys.executable, str(example_path)
        ], capture_output=True, text=True, timeout=30)
        
        # Should not have import errors (may have other errors but imports should work)
        if result.returncode != 0:
            # Check if it's an import error vs other errors
            stderr = result.stderr.lower()
            if "import" in stderr and ("error" in stderr or "failed" in stderr):
                assert False, f"Dependency import failed: {result.stderr}"
            # Other errors (like missing data files) are acceptable for dependency test


if __name__ == "__main__":
    pytest.main([__file__])
