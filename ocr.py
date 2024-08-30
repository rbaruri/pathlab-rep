import io
from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore

# Configura tus variables
project_id = "tokyo-concept-417109"
location = "us"  # Format is "us" or "eu"
processor_id = "d4996bc6d6ba788"  # Create processor before running sample
processor_version_id = "pretrained-ocr-v2.0-2023-06-02"  # Optional

def get_mime_type(file_extension: str) -> str:
    # Derive MIME type based on file extension
    file_extension = file_extension.lower()

    if file_extension == ".pdf":
        return "application/pdf"
    elif file_extension == ".png":
        return "image/png"
    elif file_extension == ".jpg" or file_extension == ".jpeg":
        return "image/jpeg"
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_document_sample(
    project_id: str,
    location: str,
    processor_id: str,
    file_content: io.BytesIO,
    mime_type: str,
    processor_version_id: Optional[str] = None,
) -> str:
    # Configure client options for a specific endpoint
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    # Create a client for the Document AI service
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if processor_version_id:
        # Construct the full resource name for the processor version
        name = client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
    else:
        # Construct the full resource name for the processor
        name = client.processor_path(project_id, location, processor_id)

    # Load binary data
    raw_document = documentai.RawDocument(content=file_content.read(), mime_type=mime_type)

    # Configure additional processing options (optional)
    process_options = documentai.ProcessOptions(
        # Process only specific pages (e.g., first page)
        individual_page_selector=documentai.ProcessOptions.IndividualPageSelector(
            pages=[1]
        )
    )

    # Configure the processing request
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document,
        process_options=process_options,
    )

    # Execute the processing request
    result = client.process_document(request=request)

    # Access the resulting document
    document = result.document

    # Return the recognized text in the document
    return document.text
