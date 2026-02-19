import google.generativeai as genai
import os
import streamlit as st

def diagnose():
    try:
        # Try to load from secrets or env
        import toml
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "r") as f:
                secrets = toml.load(f)
                api_key = secrets.get("GOOGLE_API_KEY")
        else:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            print("❌ GOOGLE_API_KEY not found in .streamlit/secrets.toml or ENV.")
            return

        genai.configure(api_key=api_key)
        
        print("--- Diagnostic Report ---")
        print(f"SDK Version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
        
        print("\nAvailable Models:")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    print(f"- {m.name} ({m.display_name})")
        except Exception as e:
            print(f"❌ Failed to list models: {e}")
            
        print("\nTesting gemini-1.5-flash:")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Hi, are you working?")
            print(f"✅ Success! Response: {response.text[:50]}...")
        except Exception as e:
            print(f"❌ Failed with gemini-1.5-flash: {e}")

        print("\nTesting gemini-1.5-pro:")
        try:
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content("Hi, are you working?")
            print(f"✅ Success! Response: {response.text[:50]}...")
        except Exception as e:
            print(f"❌ Failed with gemini-1.5-pro: {e}")

    except Exception as e:
        print(f"❌ General error: {e}")

if __name__ == "__main__":
    diagnose()
