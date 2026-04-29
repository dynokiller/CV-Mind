import requests
import json

def check_google_key(api_key):
    # Try to list models to check if the key is valid
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Success! The Google API key is working.")
            models = response.json().get('models', [])
            if models:
                print(f"Available models: {[m['name'] for m in models[:5]]}...")
            return True
        else:
            print(f"Failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    key = "AIzaSyDxDBdks16_92Ok_9TKBA_zREwwokAUW5I"
    check_google_key(key)
