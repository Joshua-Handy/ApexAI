"""
Utility to open and examine pickle files.
Useful for inspecting saved models, VecNormalize stats, or other serialized data.
"""

import pickle
import argparse
import os
import sys
import zipfile
import tempfile
from typing import Any, Dict, List


def load_pickle(filepath: str) -> Any:
    """Load data from a pickle file or zip file containing pickled data."""
    try:
        # Check if it's a zip file (like SB3 model checkpoints)
        if filepath.endswith('.zip'):
            return load_from_zip(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file '{filepath}': {e}")
        return None


def load_from_zip(zip_path: str) -> Any:
    """Load pickled data from a zip file (like SB3 model saves)."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # List contents of zip
            file_list = zip_file.namelist()
            print(f"Files in zip: {file_list}")
            
            # SB3 saves are often JSON in 'data' file, not pickle
            # Let's try to read different files and see what we can extract
            result = {}
            
            # Try reading text files first
            for filename in ['data', '_stable_baselines3_version', 'system_info.txt']:
                if filename in file_list:
                    try:
                        with zip_file.open(filename) as f:
                            content = f.read()
                            try:
                                # Try as text first
                                text_content = content.decode('utf-8')
                                result[filename] = text_content[:1000]  # First 1000 chars
                            except UnicodeDecodeError:
                                # If not text, try as pickle
                                import io
                                pickle_content = pickle.load(io.BytesIO(content))
                                result[filename] = pickle_content
                    except Exception as e:
                        result[filename] = f"Error reading: {e}"
            
            if result:
                return result
            else:
                print("Could not read any files from zip")
                return None
            
    except Exception as e:
        print(f"Error loading from zip '{zip_path}': {e}")
        return None


def safe_getattr(obj, attr, default=None):
    """Safely get attribute without triggering infinite recursion."""
    try:
        return getattr(obj, attr, default)
    except (RecursionError, AttributeError):
        return f"<RecursionError getting {attr}>"


def inspect_object(obj: Any, max_depth: int = 3, current_depth: int = 0, visited=None) -> str:
    """Recursively inspect an object's structure."""
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion by tracking visited objects
    obj_id = id(obj)
    if obj_id in visited:
        return f"{'  ' * current_depth}<circular reference to {type(obj).__name__}>"
    
    indent = "  " * current_depth
    
    if current_depth >= max_depth:
        return f"{indent}... (max depth reached)"
    
    # Add to visited set for complex objects
    if not isinstance(obj, (str, int, float, bool, type(None))):
        visited.add(obj_id)
    
    try:
        if obj is None:
            result = f"{indent}None"
        elif isinstance(obj, (str, int, float, bool)):
            result = f"{indent}{type(obj).__name__}: {obj}"
        elif isinstance(obj, (list, tuple)):
            result = f"{indent}{type(obj).__name__} (length: {len(obj)})"
            if len(obj) > 0 and current_depth < max_depth - 1:
                try:
                    result += f"\n{inspect_object(obj[0], max_depth, current_depth + 1, visited.copy())}"
                    if len(obj) > 1:
                        result += f"\n{indent}  ..."
                except (RecursionError, Exception):
                    result += f"\n{indent}  <error inspecting contents>"
        elif isinstance(obj, dict):
            result = f"{indent}dict (keys: {len(obj)})"
            if len(obj) > 0 and current_depth < max_depth - 1:
                count = 0
                for key, value in obj.items():
                    if count >= 5:  # Limit to first 5 keys
                        result += f"\n{indent}  ..."
                        break
                    try:
                        result += f"\n{indent}  '{key}': {type(value).__name__}"
                        if current_depth < max_depth - 2:
                            result += f"\n{inspect_object(value, max_depth, current_depth + 2, visited.copy())}"
                    except (RecursionError, Exception):
                        result += f"\n{indent}  '{key}': <error inspecting value>"
                    count += 1
        else:
            # For custom objects, show type and attributes safely
            result = f"{indent}{type(obj).__name__}"
            if current_depth < max_depth - 1:
                try:
                    # Try to get attributes safely
                    if hasattr(obj, '__dict__'):
                        attrs = safe_getattr(obj, '__dict__', {})
                        if isinstance(attrs, dict) and attrs:
                            result += f" (attributes: {len(attrs)})"
                            count = 0
                            for attr, value in attrs.items():
                                if count >= 5:  # Limit to first 5 attributes
                                    result += f"\n{indent}  ..."
                                    break
                                try:
                                    result += f"\n{indent}  {attr}: {type(value).__name__}"
                                except Exception:
                                    result += f"\n{indent}  {attr}: <error getting type>"
                                count += 1
                except Exception:
                    result += " <error inspecting attributes>"
        
        # Remove from visited set when done
        if not isinstance(obj, (str, int, float, bool, type(None))):
            visited.discard(obj_id)
            
        return result
        
    except Exception as e:
        if not isinstance(obj, (str, int, float, bool, type(None))):
            visited.discard(obj_id)
        return f"{indent}<error inspecting object: {e}>"


def examine_pickle(filepath: str, max_depth: int = 3, show_raw: bool = False):
    """Examine the contents of a pickle file."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return
    
    print(f"Loading pickle file: {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")
    print("-" * 50)
    
    data = load_pickle(filepath)
    if data is None:
        return
    
    print(f"Root object type: {type(data).__name__}")
    print("\nStructure:")
    print(inspect_object(data, max_depth))
    
    if show_raw:
        print("\nRaw content (first 1000 chars):")
        print(str(data)[:1000])
        if len(str(data)) > 1000:
            print("... (truncated)")
    
    # Special handling for common MetaDrive/SB3 objects
    try:
        if safe_getattr(data, 'observation_space') is not None:
            obs_space = safe_getattr(data, 'observation_space')
            print(f"\nObservation space: {obs_space}")
        if safe_getattr(data, 'action_space') is not None:
            action_space = safe_getattr(data, 'action_space')
            print(f"Action space: {action_space}")
        if (safe_getattr(data, 'running_mean') is not None and 
            safe_getattr(data, 'running_var') is not None):
            running_mean = safe_getattr(data, 'running_mean')
            running_var = safe_getattr(data, 'running_var')
            count = safe_getattr(data, 'count', 'N/A')
            print(f"\nVecNormalize stats - Running mean shape: {getattr(running_mean, 'shape', 'unknown')}")
            print(f"Running var shape: {getattr(running_var, 'shape', 'unknown')}")
            print(f"Count: {count}")
        if safe_getattr(data, 'policy') is not None:
            policy = safe_getattr(data, 'policy')
            print(f"\nModel policy type: {type(policy).__name__}")
        if safe_getattr(data, 'num_timesteps') is not None:
            num_timesteps = safe_getattr(data, 'num_timesteps')
            print(f"Training timesteps: {num_timesteps}")
    except Exception as e:
        print(f"\nError inspecting special attributes: {e}")


def main():
    parser = argparse.ArgumentParser(description="Examine pickle files")
    parser.add_argument("filepath", help="Path to the pickle file")
    parser.add_argument("--depth", "-d", type=int, default=3, 
                       help="Maximum depth for structure inspection (default: 3)")
    parser.add_argument("--raw", "-r", action="store_true",
                       help="Show raw content preview")
    parser.add_argument("--list-dir", "-l", action="store_true",
                       help="List all .pkl files in the directory of the given path")
    
    args = parser.parse_args()
    
    if args.list_dir:
        directory = os.path.dirname(args.filepath) if os.path.dirname(args.filepath) else "."
        print(f"Pickle files in '{directory}':")
        pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
        if pkl_files:
            for f in sorted(pkl_files):
                filepath = os.path.join(directory, f)
                size = os.path.getsize(filepath)
                print(f"  {f} ({size} bytes)")
        else:
            print("  No .pkl files found")
        print()
    
    examine_pickle(args.filepath, args.depth, args.raw)


if __name__ == "__main__":
    main()