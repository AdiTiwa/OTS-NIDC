#!/usr/bin/env python3
"""
Runner script for OTS-NIDC examples.
Runs all convergence tests in the examples directory.
"""

import subprocess
import sys
import os

def run_test(script_name, description):
    """Run a test script and capture output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:500])
        if result.returncode != 0:
            print(f"FAILED with return code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"ERROR running {script_name}: {e}")
        return False

def main():
    os.makedirs('figures', exist_ok=True)
    
    tests = [
        ("test_single_species.py", "Single-species OTS-NIDC (verified O(Δx⁴))"),
        ("test_coupled_same_D.py", "Coupled OTS-NIDC D1=D2 (coupling limits to O(Δx²))"),
        ("test_regime1_D1_3_D2_1.py", "Regime 1: D₁=3, D₂=1 (r=3) — needs refinement"),
        ("test_regime2_D1_1_D2_4.py", "Regime 2: D₁=1, D₂=4 (r=4) — needs refinement"),
    ]
    
    results = {}
    for script, desc in tests:
        success = run_test(script, desc)
        results[desc] = success
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for desc, success in results.items():
        status = "✓ WORKING" if success else "✗ ISSUES"
        print(f"  {status}: {desc}")
    
    all_working = all(results.values())
    print(f"\nOverall: {'ALL TESTS WORKING' if all_working else 'SOME NEED REFINEMENT'}")
    
    if os.path.exists('figures'):
        print("\nGenerated figures:")
        for f in sorted(os.listdir('figures')):
            print(f"  figures/{f}")
    
    return 0 if all_working else 1

if __name__ == "__main__":
    sys.exit(main())