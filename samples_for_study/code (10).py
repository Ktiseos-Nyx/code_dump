# In some central place or model_parsers/__init__.py
MODEL_PARSER_REGISTRY = {
    "Safetensors": SafetensorsParser,
    "GGUF": GGUFParser,
    # ... other model parser classes
}