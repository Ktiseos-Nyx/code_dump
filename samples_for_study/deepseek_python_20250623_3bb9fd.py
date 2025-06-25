def read_png_metadata(file_path):
    from PIL import PngImagePlugin
    img = Image.open(file_path)
    metadata = {}
    
    # Get standard PNG chunks
    for key, value in img.info.items():
        metadata[key] = value
    
    # Extract text chunks
    with open(file_path, 'rb') as f:
        for chunk_type, chunk_data in PngImagePlugin.getchunks(f):
            if chunk_type == b'tEXt':
                key, _, value = chunk_data.partition(b'\0')
                metadata[key.decode()] = value.decode()
    
    return metadata