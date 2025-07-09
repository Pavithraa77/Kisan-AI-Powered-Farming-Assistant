from google.adk.agents import Agent
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
import os
import base64
import io
from PIL import Image as PILImage

vertexai.init(project="cloud-llm-preview1", location="asia-southeast1")


def diagnose_plant_disease(image_data_base64: str) -> dict:
    try:
        model = GenerativeModel("gemini-1.5-pro-preview-0514")

        image_bytes = base64.b64decode(image_data_base64)
        input_image = Part.from_bytes(image_bytes, mime_type="image/jpeg")

        prompt = """
        Analyze this image of a plant.
        1. Identify if there are any signs of pest infestation or plant disease.
        2. If a pest or disease is identified, state its name clearly.
        3. Provide clear, actionable advice on how to treat or manage the issue. Focus on remedies that are
           likely to be locally available and affordable for a farmer in India (e.g., common organic solutions,
           widely available pesticides if necessary, cultural practices).
        4. If the plant appears healthy or no clear issue can be determined, state that.

        Format your response as follows:
        Diagnosis: [Disease/Pest Name or "Healthy" or "Undetermined"]
        Confidence: [High/Medium/Low or a percentage if the model provides it]
        Symptoms Observed: [List key symptoms visible in the image]
        Recommended Action: [Clear, step-by-step advice for treatment/prevention]
        Considerations for Farmers: [Practical tips, e.g., "Check local agricultural extension office for specific recommendations.", "Ensure good drainage."]
        """

        response = model.generate_content([input_image, prompt])

        if response.candidates:
            gemini_output = response.candidates[0].content.text
            return {
                "status": "success",
                "report": gemini_output
            }
        else:
            return {
                "status": "error",
                "error_message": "Gemini did not return a valid response for the image."
            }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"An error occurred during diagnosis: {e}"
        }


root_agent = Agent(
    name="crop_disease_diagnoser",
    model="gemini-2.5-flash-lite-preview-06-17",
    description="An AI assistant that helps farmers diagnose crop diseases and pests from photos and provides actionable remedies.",
    instruction="""
    I am an expert in plant pathology and pest management.
    A farmer can send me a photo of a diseased plant, and I will analyze it to identify the problem
    and provide practical, actionable advice on how to address it, keeping in mind local and affordable solutions
    for farmers in India.

    Please provide a clear photo of the affected plant part.
    """,
    tools=[diagnose_plant_disease]
)
