
import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

async def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment!")
        return

    print(f"Checking Gemini API Key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        genai.configure(api_key=api_key)
        
        print("Listing available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
        
        model_name = 'gemini-2.5-flash'
        print(f"\nTesting with {model_name}...")
        model = genai.GenerativeModel(model_name)
        
        print("Sending test request...")
        response = await asyncio.to_thread(model.generate_content, "Say 'Hello from Gemini!'")
        
        if response and response.text:
            print(f"✅ Success! Response: {response.text}")
        else:
            print("❌ Failed: Empty response")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini())
