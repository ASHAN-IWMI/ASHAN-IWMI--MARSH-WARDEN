import google.generativeai as genai
import os
import re

def get_key():
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        with open(secrets_path, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r'GOOGLE_API_KEY\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    return os.getenv("GOOGLE_API_KEY")

def list_models():
    api_key = get_key()
    output = []
    if not api_key:
        output.append("GOOGLE_API_KEY not found")
    else:
        try:
            genai.configure(api_key=api_key)
            output.append("Available models:")
            for m in genai.list_models():
                output.append(f"- {m.name}")
        except Exception as e:
            output.append(f"Error: {e}")
    
    with open("model_list_utf8.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    list_models()
