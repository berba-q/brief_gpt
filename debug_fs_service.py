"""
FAOSTAT Service Debugger

This script helps debug issues with the FAOSTAT service by testing
the API endpoints directly and showing the exact data structure returned.
"""

import streamlit as st
import requests
import json
import traceback
import pandas as pd
from typing import Dict, List, Any

def debug_faostat_service():
    """Main debugging interface for FAOSTAT service."""
    
    st.title("🔧 FAOSTAT Service Debugger")
    st.markdown("Debug and test the FAOSTAT API endpoints directly")
    
    # Test configuration
    st.subheader("📡 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_url = st.text_input(
            "Base URL",
            value="https://bulks-faostat.fao.org/production/",
            help="FAOSTAT API base URL"
        )
    
    with col2:
        endpoint = st.text_input(
            "Datasets Endpoint", 
            value="datasets_E.json",
            help="Endpoint for fetching datasets list"
        )
    
    full_url = f"{base_url}{endpoint}"
    st.code(f"Full URL: {full_url}")
    
    # Test buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌐 Test Raw Connection", use_container_width=True):
            test_raw_connection(full_url)
    
    with col2:
        if st.button("🔍 Analyze Response Structure", use_container_width=True):
            analyze_response_structure(full_url)
    
    with col3:
        if st.button("🛠️ Test Service Processing", use_container_width=True):
            test_service_processing(full_url)

def test_raw_connection(url: str):
    """Test the basic HTTP connection to FAOSTAT."""
    
    st.subheader("🌐 Raw Connection Test")
    
    try:
        with st.spinner("Making HTTP request..."):
            response = requests.get(url, timeout=30)
        
        # Basic response info
        st.success(f"✅ HTTP {response.status_code}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Response Info:**")
            st.write(f"Status: {response.status_code}")
            st.write(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
            st.write(f"Content Length: {len(response.content)} bytes")
            st.write(f"Encoding: {response.encoding}")
        
        with col2:
            st.markdown("**Response Headers:**")
            for key, value in response.headers.items():
                st.write(f"{key}: {value}")
        
        # Raw content preview
        st.markdown("**Raw Response Preview (first 1000 chars):**")
        st.code(response.text[:1000] + ("..." if len(response.text) > 1000 else ""))
        
        # Try to parse as JSON
        try:
            data = response.json()
            st.success(f"✅ Valid JSON - Type: {type(data)}")
            
            if isinstance(data, list):
                st.write(f"JSON List Length: {len(data)}")
            elif isinstance(data, dict):
                st.write(f"JSON Dict Keys: {list(data.keys())}")
            else:
                st.write(f"JSON Type: {type(data)}")
            
            return data
            
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {str(e)}")
            st.write("Response is not valid JSON")
            return None
    
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out (30 seconds)")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error - check URL and internet connection")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.code(traceback.format_exc())
        return None

def analyze_response_structure(url: str):
    """Analyze the detailed structure of the FAOSTAT response."""
    
    st.subheader("🔍 Response Structure Analysis")
    
    try:
        with st.spinner("Fetching and analyzing data..."):
            response = requests.get(url, timeout=30)
            data = response.json()
        
        st.success("✅ Data fetched successfully")
        
        # Type analysis
        st.markdown("### 📊 Data Type Analysis")
        st.write(f"**Root Type:** {type(data)}")
        
        if isinstance(data, list):
            st.write(f"**List Length:** {len(data)}")
            
            if len(data) > 0:
                st.markdown("### 🔍 First Item Analysis")
                first_item = data[0]
                st.write(f"**First Item Type:** {type(first_item)}")
                
                if isinstance(first_item, dict):
                    st.markdown("**First Item Keys:**")
                    for key in first_item.keys():
                        value = first_item[key]
                        st.write(f"- `{key}`: {type(value).__name__} = {repr(value)[:100]}...")
                
                # Show first few items
                st.markdown("### 📋 Sample Items")
                for i, item in enumerate(data[:3]):
                    with st.expander(f"Item {i+1}"):
                        if isinstance(item, dict):
                            for key, value in item.items():
                                st.write(f"**{key}:** {repr(value)}")
                        else:
                            st.write(f"**Value:** {repr(item)}")
                
                # Field analysis across all items
                st.markdown("### 🏷️ Field Analysis Across All Items")
                if all(isinstance(item, dict) for item in data):
                    all_keys = set()
                    for item in data:
                        all_keys.update(item.keys())
                    
                    st.write(f"**Total Unique Fields:** {len(all_keys)}")
                    st.write("**All Fields Found:**")
                    for key in sorted(all_keys):
                        # Count how many items have this field
                        count = sum(1 for item in data if key in item)
                        st.write(f"- `{key}` (present in {count}/{len(data)} items)")
            else:
                st.warning("List is empty")
        
        elif isinstance(data, dict):
            st.write(f"**Dictionary Keys:** {list(data.keys())}")
            
            st.markdown("### 📋 Dictionary Contents")
            for key, value in data.items():
                st.write(f"**{key}:** {type(value).__name__}")
                if isinstance(value, (str, int, float)):
                    st.write(f"  Value: {repr(value)}")
                elif isinstance(value, list):
                    st.write(f"  Length: {len(value)}")
                    if len(value) > 0:
                        st.write(f"  First item: {repr(value[0])[:100]}...")
        
        else:
            st.write(f"**Unexpected data type:** {type(data)}")
            st.write(f"**Content:** {repr(data)[:500]}...")
        
        return data
    
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.code(traceback.format_exc())
        return None

def test_service_processing(url: str):
    """Test how the FAOSTAT service would process this data."""
    
    st.subheader("🛠️ Service Processing Test")
    
    try:
        # Fetch the raw data
        with st.spinner("Fetching data..."):
            response = requests.get(url, timeout=30)
            response_data = response.json()
        
        st.success("✅ Raw data fetched")
        st.write(f"Raw data type: {type(response_data)}")
        
        # UPDATED: Use the same logic as the fixed service
        st.markdown("### 🔄 Simulating Fixed Service Processing")
        
        # Handle the complex nested structure (same as fixed service)
        datasets = None
        
        if isinstance(response_data, dict) and "Datasets" in response_data:
            datasets_container = response_data["Datasets"]
            st.write(f"Found Datasets container: {type(datasets_container)}")
            
            if isinstance(datasets_container, dict):
                if "Dataset" in datasets_container:
                    # Standard case: datasets at ["Datasets"]["Dataset"]
                    datasets = datasets_container["Dataset"]
                    st.success(f"✅ Found datasets at ['Datasets']['Dataset']: {type(datasets)}")
                else:
                    st.warning("⚠️ No 'Dataset' key found in Datasets container")
                    st.write(f"Available keys: {list(datasets_container.keys())}")
            elif isinstance(datasets_container, list):
                # Direct list case
                datasets = datasets_container
                st.success(f"✅ Datasets container is direct list: {len(datasets)} items")
        else:
            st.error("❌ No 'Datasets' key found in response")
            st.write(f"Available keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
        
        if datasets is None:
            st.error("❌ Could not find datasets in response")
            return None
        
        # Normalize to list format (same as fixed service)
        if isinstance(datasets, dict):
            # Single dataset - convert to list
            st.info("ℹ️ Single dataset found, converting to list")
            datasets = [datasets]
        elif isinstance(datasets, list):
            # Multiple datasets - use as is
            st.success(f"✅ Multiple datasets found: {len(datasets)} items")
        else:
            st.error(f"❌ Unexpected datasets type: {type(datasets)}")
            return None
        
        # Validate that we have a list of dictionaries
        if not all(isinstance(item, dict) for item in datasets):
            st.error("❌ Not all dataset items are dictionaries")
            st.write(f"Item types: {[type(item) for item in datasets[:3]]}")
            return None
        
        st.success(f"✅ Validation passed: {len(datasets)} dataset dictionaries")
        
        # Process the datasets (same field names as fixed service)
        st.markdown("### 📊 Processing Individual Datasets")
        
        datasets_info = []
        
        for i, dataset in enumerate(datasets[:5]):  # Process first 5 for debugging
            st.write(f"\n**Processing dataset {i+1}:**")
            st.write(f"Dataset keys: {list(dataset.keys())}")
            
            # Use the same field extraction logic as the fixed service
            name = (dataset.get('DatasetName') or 
                   dataset.get('name') or 
                   dataset.get('Name') or 
                   dataset.get('title') or
                   dataset.get('Title') or
                   'Unknown')
            
            code = (dataset.get('DatasetCode') or 
                   dataset.get('code') or 
                   dataset.get('Code') or 
                   dataset.get('id') or
                   dataset.get('ID') or
                   f"dataset_{i}")
            
            description = (dataset.get('DatasetDescription') or 
                         dataset.get('description') or 
                         dataset.get('Description') or 
                         dataset.get('Topic') or
                         'No description available')
            
            st.write(f"- **Name:** {name}")
            st.write(f"- **Code:** {code}")
            st.write(f"- **Description:** {description[:100]}...")
            
            # Only add if we have at least a name or code
            if name != 'Unknown' or (code != f"dataset_{i}" and code != 'Unknown'):
                processed_item = {
                    'code': code,
                    'name': name,
                    'description': description,
                    'metadata': dataset
                }
                datasets_info.append(processed_item)
                st.success(f"✅ Dataset {i+1} processed successfully")
            else:
                st.warning(f"⚠️ Dataset {i+1} skipped - no recognizable name or code")
        
        # Results summary
        st.markdown("### 📊 Final Results")
        st.write(f"**Total datasets in response:** {len(datasets)}")
        st.write(f"**Successfully processed:** {len(datasets_info)}")
        
        if datasets_info:
            st.success("✅ Processing successful!")
            
            # Convert to DataFrame
            df = pd.DataFrame(datasets_info)
            st.write(f"**DataFrame shape:** {df.shape}")
            st.write("**DataFrame columns:**", list(df.columns))
            
            st.markdown("**Sample processed data:**")
            st.dataframe(df)
            
            # Test the actual service
            st.markdown("### 🎯 Testing Your Actual Service")
            if st.button("🔄 Test Real Service"):
                test_real_service()
            
        else:
            st.error("❌ No datasets were processed successfully")
        
        return datasets_info
    
    except Exception as e:
        st.error(f"❌ Processing test failed: {str(e)}")
        st.code(traceback.format_exc())
        return None

def test_real_service():
    """Test the actual FAOSTAT service from session state."""
    
    st.markdown("#### 🎯 Real Service Test")
    
    try:
        if 'faostat_service' in st.session_state:
            service = st.session_state.faostat_service
            
            with st.spinner("Testing real service..."):
                df = service.get_available_datasets(force_refresh=True)
            
            if df is not None and not df.empty:
                st.success(f"✅ Real service works! Found {len(df)} datasets")
                st.dataframe(df.head())
            else:
                st.error("❌ Real service returned empty results")
                st.write("Check the application logs for detailed error messages")
        else:
            st.error("❌ FAOSTAT service not found in session state")
            
    except Exception as e:
        st.error(f"❌ Real service test failed: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    debug_faostat_service()