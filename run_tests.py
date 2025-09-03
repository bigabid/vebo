#!/usr/bin/env python3
"""
Test runner script for Vebo profiler tests.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=True, parallel=False):
    """
    Run tests with specified options.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Whether to run in verbose mode
        coverage: Whether to generate coverage report
        parallel: Whether to run tests in parallel
    """
    
    # Base pytest command
    cmd = ["python3", "-m", "pytest"]
    
    # Add test path based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Add options
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=python/vebo_profiler",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ])
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add additional options
    cmd.extend([
        "--tb=short",
        "--maxfail=10",
        "--durations=10"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        
        if coverage:
            print("\n📊 Coverage report generated:")
            print("   - HTML: htmlcov/index.html")
            print("   - XML: coverage.xml")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return False


def install_test_dependencies():
    """Install test dependencies."""
    print("Installing test dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "-r", "python/requirements.txt"
        ], check=True)
        print("✅ Test dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install test dependencies")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Vebo profiler tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Run in verbose mode"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--parallel", "-p", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install test dependencies first"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🧪 Vebo Profiler Test Runner")
    print("=" * 50)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            return 1
    
    # Run tests
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=not args.no_coverage,
        parallel=args.parallel
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
