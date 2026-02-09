# import os
# import json
# from google import genai
# from google.genai import types
# from typing import List, Dict
# from dotenv import load_dotenv
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_exponential,
#     retry_if_exception_type
# )
# load_dotenv()

# class StructuredExtractor:
#     def __init__(self):
#         self.client = genai.Client()
#         # Ensure you are using the correct model version
#         self.model_name = "gemini-3-flash-preview" 

#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_exponential(multiplier=1, min=2, max=10),
#         retry=retry_if_exception_type(Exception) 
#     )
#     def extract(self, document_chunks: List[Dict]) -> Dict:
#         try:
#             full_text = "\n\n".join([chunk['content'] for chunk in document_chunks])
            
#             # Define the expected JSON schema
#             response_schema = {
#                 "type": "OBJECT",
#                 "properties": {
#                     "shipment_id": {"type": "STRING"},
#                     "shipper": {"type": "STRING"},
#                     "consignee": {"type": "STRING"},
#                     "pickup_datetime": {"type": "STRING"},
#                     "delivery_datetime": {"type": "STRING"},
#                     "equipment_type": {"type": "STRING"},
#                     "mode": {"type": "STRING"},
#                     "rate": {"type": "NUMBER"},
#                     "currency": {"type": "STRING"},
#                     "weight": {"type": "STRING"},
#                     "carrier_name": {"type": "STRING"},
#                 },
#                 "required": ["shipment_id", "shipper", "consignee"]
#             }

#             system_prompt = """
#             You are a logistics data extractor. Extract information from the provided document.
#             Mapping Guidelines:
#             - shipment_id: Use 'Reference ID'.
#             - shipper: The origin/pickup company name (e.g., AAA Los Angeles)[cite: 16].
#             - consignee: The destination/drop company name (e.g., xyz)[cite: 29].
#             - equipment_type: The trailer type mentioned (e.g., Flatbed).
#             - mode: The load type (e.g., FTL)[cite: 18].
#             - rate: The numeric value of the agreed amount[cite: 9].
#             - currency: The currency code (e.g., USD)[cite: 9].
#             - Use null for missing data.
#             """

#             response = self.client.models.generate_content(
#                 model=self.model_name,
#                 contents=full_text,
#                 config=types.GenerateContentConfig(
#                     system_instruction=system_prompt,
#                     response_mime_type="application/json",
#                     response_schema=response_schema,
#                     temperature=0.0,
#                 )
#             )
            
#             # The SDK returns the parsed JSON object directly when response_schema is used
#             return response.parsed
#         except Exception as e:
#             raise e

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from pydantic import BaseModel, Field

load_dotenv()

# Pydantic schema for structured output
class ShipmentData(BaseModel):
    shipment_id: str | None = Field(None, description="Reference ID")
    shipper: str | None = Field(None, description="Origin/pickup company name")
    consignee: str | None = Field(None, description="Destination/drop company name")
    pickup_datetime: str | None = Field(None, description="Pickup date/time")
    delivery_datetime: str | None = Field(None, description="Delivery date/time")
    equipment_type: str | None = Field(None, description="Trailer type (Flatbed, etc.)")
    mode: str | None = Field(None, description="Load type (FTL, LTL)")
    rate: float | None = Field(None, description="Numeric rate value")
    currency: str | None = Field(None, description="Currency code (USD)")
    weight: str | None = Field(None, description="Shipment weight")
    carrier_name: str | None = Field(None, description="Carrier company name")

class StructuredExtractor:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0.0,
            num_predict=1000
        )
        
        self.extraction_prompt = PromptTemplate(
            template="""You are a logistics data extractor. Extract information from the document below.

Mapping Guidelines:
- shipment_id: Use 'Reference ID' or similar identifier
- shipper: Origin/pickup company name (e.g., "AAA Los Angeles")  
- consignee: Destination/drop company name (e.g., "XYZ Distribution")
- equipment_type: Trailer type (e.g., "Flatbed", "Reefer", "Dry Van")
- mode: Load type (e.g., "FTL", "LTL", "Partial")
- rate: Numeric value of agreed amount ONLY (e.g., 2500.00)
- currency: Currency code (e.g., "USD", "EUR")
- pickup_datetime: Pickup date/time in ISO format or readable format
- delivery_datetime: Delivery date/time 
- weight: Total shipment weight with units
- carrier_name: Carrier company name
- Use null for missing data

Document:
{document}

Respond with ONLY valid JSON matching this schema:""",
            input_variables=["document"]
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def extract(self, document_chunks: List[Dict]) -> Dict:
        try:
            print("Extraction started")
            full_text = "\n\n".join([chunk['content'] for chunk in document_chunks])
            
            chain = self.extraction_prompt | self.llm | JsonOutputParser(pydantic_object=ShipmentData)
            
            result = chain.invoke({"document": full_text})
            print("Extraction ended")
            print(result)
            return result
            
        except Exception as e:
            print(f"Extraction failed: {e}")
            return {
                "shipment_id": None, "shipper": None, "consignee": None,
                "pickup_datetime": None, "delivery_datetime": None,
                "equipment_type": None, "mode": None, "rate": None,
                "currency": None, "weight": None, "carrier_name": None
            }
