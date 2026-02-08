import os
import json
from google import genai
from google.genai import types
from typing import List, Dict
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
load_dotenv()

class StructuredExtractor:
    def __init__(self):
        self.client = genai.Client()
        # Ensure you are using the correct model version
        self.model_name = "gemini-3-flash-preview" 

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception) 
    )
    def extract(self, document_chunks: List[Dict]) -> Dict:
        try:
            full_text = "\n\n".join([chunk['content'] for chunk in document_chunks])
            
            # Define the expected JSON schema
            response_schema = {
                "type": "OBJECT",
                "properties": {
                    "shipment_id": {"type": "STRING"},
                    "shipper": {"type": "STRING"},
                    "consignee": {"type": "STRING"},
                    "pickup_datetime": {"type": "STRING"},
                    "delivery_datetime": {"type": "STRING"},
                    "equipment_type": {"type": "STRING"},
                    "mode": {"type": "STRING"},
                    "rate": {"type": "NUMBER"},
                    "currency": {"type": "STRING"},
                    "weight": {"type": "STRING"},
                    "carrier_name": {"type": "STRING"},
                },
                "required": ["shipment_id", "shipper", "consignee"]
            }

            system_prompt = """
            You are a logistics data extractor. Extract information from the provided document.
            Mapping Guidelines:
            - shipment_id: Use 'Reference ID'.
            - shipper: The origin/pickup company name (e.g., AAA Los Angeles)[cite: 16].
            - consignee: The destination/drop company name (e.g., xyz)[cite: 29].
            - equipment_type: The trailer type mentioned (e.g., Flatbed).
            - mode: The load type (e.g., FTL)[cite: 18].
            - rate: The numeric value of the agreed amount[cite: 9].
            - currency: The currency code (e.g., USD)[cite: 9].
            - Use null for missing data.
            """

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.0,
                )
            )
            
            # The SDK returns the parsed JSON object directly when response_schema is used
            return response.parsed
        except Exception as e:
            raise e
