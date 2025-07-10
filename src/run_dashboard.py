"""
Script to run the customer segmentation dashboard.
"""

import os
import sys

# Add the current directory to the path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# Also add the parent directory to ensure all imports work correctly
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the dashboard module
from dashboard import run_dashboard

if __name__ == "__main__":
    # Print working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Run the dashboard
    run_dashboard()
