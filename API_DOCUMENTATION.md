# PropertyData API Documentation

The PropertyData API provides two main services:
1. **Property Enrichment**: Real estate data scraping and enrichment
2. **Document Classification**: AI-powered document type classification for loan collateral

## Base URL
- **Local Development**: `http://localhost:5000`
- **Production**: Your deployed service URL

## Authentication
Currently no authentication required.

## Endpoints

### 1. Health Check
**GET** `/`

Check if the API is running and get service status.

**Response:**
```json
{
  "status": "ok",
  "message": "PropertyData API is running.",
  "document_classification_model": "loaded" | "not_loaded"
}
```

### 2. API Information
**GET** `/api-info`

Get comprehensive information about all available services and endpoints.

**Response:**
```json
{
  "service_name": "PropertyData API",
  "version": "1.0",
  "services": {
    "property_enrichment": {
      "status": "available",
      "endpoints": ["/enrich", "/search-options"],
      "description": "Real estate property data enrichment using Realtor.com scraping"
    },
    "document_classification": {
      "status": "available",
      "model_status": "loaded",
      "endpoints": ["/predict", "/model-info"],
      "description": "Document classification for loan collateral analysis"
    }
  },
  "endpoints": {
    "/": "Health check",
    "/enrich": "POST - Enrich property data by address",
    "/search-options": "GET - Get property search options",
    "/predict": "POST - Classify PDF document pages",
    "/model-info": "GET - Get document classification model information",
    "/api-info": "GET - Get API information"
  }
}
```

## Property Enrichment Endpoints

### 3. Enrich Property Data
**POST** `/enrich`

Scrape and enrich property data from Realtor.com.

**Request Body:**
```json
{
  "address": "Dallas, TX",
  "listing_type": "for_sale",
  "past_days": 30,
  "radius": 1.0,
  "mls_only": false,
  "limit": 100
}
```

**Parameters:**
- `address` (required): Location to search
- `listing_type` (optional): "for_sale", "for_rent", "sold", "pending"
- `past_days` (optional): Get properties from last N days
- `radius` (optional): Search radius in miles
- `mls_only` (optional): Only fetch MLS listings
- `limit` (optional): Maximum results (default: 100, max: 10000)

**Response:**
```json
{
  "success": true,
  "count": 25,
  "properties": [
    {
      "property_url": "...",
      "property_id": "...",
      "address": "...",
      "price": 450000,
      "beds": 3,
      "baths": 2,
      "sqft": 1800,
      ...
    }
  ]
}
```

### 4. Get Search Options
**GET** `/search-options`

Get available options for property searches.

**Response:**
```json
{
  "listing_types": ["for_sale", "for_rent", "sold", "pending"],
  "property_types": ["single_family", "multi_family", "condos", ...],
  "parameters": {
    "address": "Required. Location to search",
    "listing_type": "Optional. Type of listing",
    ...
  }
}
```

## Document Classification Endpoints

### 5. Predict Document Type
**POST** `/predict`

Upload a PDF and get AI-powered classification for each page.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: PDF file in `file` field

**cURL Example:**
```bash
curl -X POST \
  http://localhost:5000/predict \
  -F "file=@path/to/document.pdf"
```

**Python Example:**
```python
import requests

with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    response = requests.post('http://localhost:5000/predict', files=files)
    result = response.json()
```

**Response:**
```json
{
  "success": true,
  "filename": "mortgage_docs.pdf",
  "page_count": 5,
  "predictions": [
    {
      "page": 1,
      "predicted_label": "Note",
      "confidence": 0.95,
      "text_length": 1250
    },
    {
      "page": 2,
      "predicted_label": "Mortgage",
      "confidence": 0.87,
      "text_length": 2100
    },
    ...
  ]
}
```

### 6. Get Model Information
**GET** `/model-info`

Get information about the loaded document classification model.

**Response:**
```json
{
  "model_loaded": true,
  "model_type": "<class 'sklearn.pipeline.Pipeline'>",
  "model_path": "projects/doc_classifier_model.joblib",
  "accuracy": 0.92,
  "training_samples": 450,
  "test_samples": 113,
  "supported_labels": ["Note", "Mortgage", "Assignment", "Deed of Trust", ...],
  "min_accuracy_threshold": 0.85
}
```

## Document Types Supported

The document classification model can identify these loan document types:

- **Note**: Promissory notes
- **Mortgage**: Mortgage documents
- **Deed of Trust**: Deed of trust documents
- **Assignment**: Assignment of mortgage/deed of trust
- **Allonge**: Allonge documents
- **Rider**: Riders and addendums
- **Bailee Letter**: Bailee letters
- **UNLABELED**: Documents that don't match known patterns

## Error Responses

All endpoints return error responses in this format:

```json
{
  "error": "Error description",
  "details": "Additional error details (if available)"
}
```

Common HTTP status codes:
- `400`: Bad Request (missing parameters, invalid file)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Testing the API

Use the provided test script:

```bash
# Test all endpoints
python projects/test_api.py

# Test with a specific PDF
python projects/test_api.py path/to/document.pdf
```

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider adding rate limiting middleware.

## Security Considerations

- PDF files are temporarily stored and then deleted
- No authentication is currently required
- Consider adding API keys or other authentication for production use
- Validate file types and sizes to prevent abuse

## Dependencies

The API requires these system dependencies:
- **Tesseract OCR**: For text extraction from PDFs
- **Poppler**: For PDF to image conversion

Install on different systems:
- **macOS**: `brew install tesseract poppler`
- **Ubuntu**: `sudo apt-get install tesseract-ocr poppler-utils`
- **Windows**: Download and install from official sources