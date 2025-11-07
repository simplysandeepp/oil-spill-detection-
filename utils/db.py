# ============================================================================
# DB.PY - Supabase Database Connection and Helper Functions (FIXED VERSION)
# ============================================================================

from supabase import create_client, Client
import os
import io
from PIL import Image
from datetime import datetime
import numpy as np

# Initialize Supabase client
supabase: Client = None

# Try loading Streamlit secrets first (for Streamlit Cloud deployment)
try:
    import streamlit as st
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    print("✅ Loaded Supabase credentials from Streamlit secrets")
except (ImportError, KeyError, FileNotFoundError) as e:
    print(f"⚠️ Could not load from Streamlit secrets: {e}")
    # Fallback to .env file (for local development)
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        load_dotenv(dotenv_path)
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        print(f"✅ Loaded Supabase credentials from .env file at {dotenv_path}")
    except Exception as env_error:
        print(f"⚠️ Could not load from .env: {env_error}")
        SUPABASE_URL = None
        SUPABASE_KEY = None

# Initialize Supabase client if credentials exist
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")
        print(f"   URL: {SUPABASE_URL}")
    except Exception as e:
        supabase = None
        print(f"❌ Failed to initialize Supabase client: {e}")
else:
    supabase = None
    print("⚠️ Supabase credentials not found!")


DEFAULT_TABLE = "oil_detections"
STORAGE_BUCKET = "detection-images"  # Your Supabase storage bucket name


# ============================================================================
# IMAGE STORAGE FUNCTIONS (FIXED)
# ============================================================================

def ensure_pil_image(img):
    """Convert numpy array or ensure PIL Image"""
    if isinstance(img, np.ndarray):
        # Ensure uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        return Image.fromarray(img)
    return img


def save_images_to_storage(filename, overlay_img, heatmap_img, binary_mask_img):
    """
    Save detection images to Supabase Storage - FIXED VERSION
    
    Returns:
        dict: URLs of saved images {'overlay': url, 'heatmap': url, 'binary_mask': url}
        None: If upload failed completely
    """
    if supabase is None:
        print("⚠️ Supabase client not initialized - cannot save images")
        return None
    
    try:
        # Create unique folder name with milliseconds for better uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = filename.rsplit('.', 1)[0].replace(' ', '_').replace('(', '').replace(')', '')
        folder = f"{timestamp}_{base_name}"
        
        images_data = {
            'overlay': overlay_img,
            'heatmap': heatmap_img,
            'binary_mask': binary_mask_img
        }
        
        urls = {}
        upload_errors = []
        
        for img_type, img_data in images_data.items():
            try:
                # Convert to PIL Image
                pil_img = ensure_pil_image(img_data)
                
                # Convert to bytes
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format='PNG', optimize=True)
                img_bytes.seek(0)
                img_data_bytes = img_bytes.getvalue()
                
                # Define storage path
                file_path = f"{folder}/{img_type}.png"
                
                print(f"📤 Uploading {img_type} to: {file_path} (size: {len(img_data_bytes)} bytes)")
                
                # Upload to Supabase Storage with proper error handling
                try:
                    response = supabase.storage.from_(STORAGE_BUCKET).upload(
                        path=file_path,
                        file=img_data_bytes,
                        file_options={
                            "content-type": "image/png",
                            "upsert": "false"
                        }
                    )
                    
                    # Check if response indicates success
                    if response:
                        # Get public URL
                        public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(file_path)
                        urls[img_type] = public_url
                        print(f"✅ Successfully uploaded {img_type}: {public_url}")
                    else:
                        urls[img_type] = ""
                        error_msg = f"Upload returned empty response for {img_type}"
                        upload_errors.append(error_msg)
                        print(f"❌ {error_msg}")
                        
                except Exception as upload_error:
                    urls[img_type] = ""
                    error_msg = f"Upload failed for {img_type}: {str(upload_error)}"
                    upload_errors.append(error_msg)
                    print(f"❌ {error_msg}")
                
            except Exception as img_error:
                urls[img_type] = ""
                error_msg = f"Image processing failed for {img_type}: {str(img_error)}"
                upload_errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        # If all uploads failed, return None
        if all(url == "" for url in urls.values()):
            print(f"❌ All image uploads failed. Errors: {upload_errors}")
            return None
        
        # Return URLs (some may be empty strings if failed)
        if upload_errors:
            print(f"⚠️ Some uploads had errors: {upload_errors}")
        
        return urls
    
    except Exception as e:
        print(f"❌ Critical error in save_images_to_storage: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def insert_detection_data(data: dict, table_name: str = DEFAULT_TABLE):
    """
    Insert a detection record into the Supabase table.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📤 Inserting data into table '{table_name}':")
        print(f"   Data keys: {list(data.keys())}")
        print(f"   Has URLs: overlay={bool(data.get('overlay_url'))}, heatmap={bool(data.get('heatmap_url'))}, binary={bool(data.get('binary_mask_url'))}")
        
        response = supabase.table(table_name).insert(data).execute()
        
        print(f"✅ Successfully inserted record into '{table_name}'")
        
        return response
    
    except Exception as e:
        print(f"❌ Error inserting data into '{table_name}': {e}")
        import traceback
        traceback.print_exc()
        raise


def fetch_all_detections(table_name: str = DEFAULT_TABLE):
    """
    Fetch all detection records from the Supabase table.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📥 Fetching all records from table '{table_name}'...")
        
        response = supabase.table(table_name).select("*").order('timestamp', desc=True).execute()
        
        print(f"✅ Successfully fetched {len(response.data)} records from '{table_name}'")
        
        return response.data
    
    except Exception as e:
        print(f"❌ Error fetching data from '{table_name}': {e}")
        raise


def fetch_recent_detections(table_name: str = DEFAULT_TABLE, limit: int = 10):
    """
    Fetch recent detection records from the Supabase table.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📥 Fetching {limit} recent records from table '{table_name}'...")
        
        response = supabase.table(table_name).select("*").order('timestamp', desc=True).limit(limit).execute()
        
        print(f"✅ Successfully fetched {len(response.data)} recent records from '{table_name}'")
        
        return response.data
    
    except Exception as e:
        print(f"❌ Error fetching recent data from '{table_name}': {e}")
        raise


def fetch_detection_with_images(detection_id: int, table_name: str = DEFAULT_TABLE):
    """
    Fetch a specific detection record with all image URLs.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📥 Fetching detection {detection_id} from table '{table_name}'...")
        
        response = supabase.table(table_name).select("*").eq('id', detection_id).execute()
        
        if response.data:
            print(f"✅ Successfully fetched detection {detection_id}")
            return response.data[0]
        else:
            print(f"⚠️ No detection found with id {detection_id}")
            return None
    
    except Exception as e:
        print(f"❌ Error fetching detection {detection_id}: {e}")
        raise


def fetch_spill_detections_only(table_name: str = DEFAULT_TABLE):
    """
    Fetch only records where oil spills were detected.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📥 Fetching spill detections from table '{table_name}'...")
        
        response = supabase.table(table_name).select("*").eq('has_spill', True).order('timestamp', desc=True).execute()
        
        print(f"✅ Successfully fetched {len(response.data)} spill detections from '{table_name}'")
        
        return response.data
    
    except Exception as e:
        print(f"❌ Error fetching spill detections from '{table_name}': {e}")
        raise


def delete_detection(record_id: int, table_name: str = DEFAULT_TABLE):
    """
    Delete a specific detection record by ID.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"🗑️ Deleting record {record_id} from table '{table_name}'...")
        
        response = supabase.table(table_name).delete().eq('id', record_id).execute()
        
        print(f"✅ Successfully deleted record {record_id} from '{table_name}'")
        
        return response
    
    except Exception as e:
        print(f"❌ Error deleting record {record_id} from '{table_name}': {e}")
        raise


def get_database_stats(table_name: str = DEFAULT_TABLE):
    """
    Get statistics about the database records.
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📊 Calculating statistics for table '{table_name}'...")
        
        # Fetch all records
        all_records = fetch_all_detections(table_name)
        
        if not all_records:
            return {
                'total_records': 0,
                'spills_detected': 0,
                'clean_images': 0,
                'avg_coverage': 0.0,
                'avg_confidence': 0.0,
                'detection_rate': 0.0
            }
        
        # Calculate statistics
        total_records = len(all_records)
        spills_detected = sum(1 for r in all_records if r.get('has_spill', False))
        clean_images = total_records - spills_detected
        
        # Calculate averages
        avg_coverage = sum(r.get('coverage_percentage', 0) for r in all_records) / total_records
        avg_confidence = sum(r.get('avg_confidence', 0) for r in all_records) / total_records
        detection_rate = (spills_detected / total_records * 100) if total_records > 0 else 0.0
        
        stats = {
            'total_records': total_records,
            'spills_detected': spills_detected,
            'clean_images': clean_images,
            'avg_coverage': round(avg_coverage, 2),
            'avg_confidence': round(avg_confidence, 3),
            'detection_rate': round(detection_rate, 2)
        }
        
        print(f"✅ Statistics calculated: {stats}")
        
        return stats
    
    except Exception as e:
        print(f"❌ Error calculating statistics for '{table_name}': {e}")
        raise


def test_connection():
    """
    Test the Supabase database connection.
    """
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        print("🔍 Testing Supabase connection...")
        
        # Try to fetch one record to test connection
        response = supabase.table(DEFAULT_TABLE).select("*").limit(1).execute()
        
        print("✅ Supabase connection successful!")
        print(f"   Table '{DEFAULT_TABLE}' is accessible")
        
        return True
    
    except Exception as e:
        print(f"❌ Supabase connection test failed: {str(e)}")
        return False


def test_storage_connection():
    """
    Test the Supabase storage connection.
    """
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        print(f"🔍 Testing Supabase storage bucket '{STORAGE_BUCKET}'...")
        
        # Try to list files in the bucket
        response = supabase.storage.from_(STORAGE_BUCKET).list()
        
        print(f"✅ Storage bucket '{STORAGE_BUCKET}' is accessible!")
        print(f"   Files in bucket: {len(response) if response else 0}")
        
        return True
    
    except Exception as e:
        print(f"❌ Storage test failed: {str(e)}")
        print(f"   Make sure bucket '{STORAGE_BUCKET}' exists and is public")
        return False


# Test connection when module is imported
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUPABASE DATABASE CONNECTION TEST")
    print("="*60 + "\n")
    
    if test_connection():
        print("\n✅ Database connection test passed!")
        
        # Test storage
        if test_storage_connection():
            print("\n✅ Storage connection test passed!")
        
        # Try to get stats
        try:
            stats = get_database_stats()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"\n⚠️ Could not fetch statistics: {str(e)}")
    else:
        print("\n❌ Database connection test failed!")
        print("\nTroubleshooting:")
        print("1. Check if SUPABASE_URL and SUPABASE_KEY are set correctly")
        print("2. Verify your Supabase project is active")
        print("3. Ensure the 'oil_detections' table exists in your database")
        print("4. Check if your API key has the correct permissions")