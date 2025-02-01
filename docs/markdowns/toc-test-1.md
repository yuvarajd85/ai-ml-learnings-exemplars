Identifying a file format in Python can be achieved by examining the file's content or its extension. Here are several approaches:

---

----

### **1. Using the mimetypes Module**
The `mimetypes` module maps file extensions to MIME types.
```python
import mimetypes

file_path = "example.pkl"
mime_type, _ = mimetypes.guess_type(file_path)
print(f"MIME type: {mime_type}")
```
- **Limitation**: Relies on file extensions and may fail for files with incorrect or missing extensions.

**[⬆ back to top](#table-of-contents)**

---

### **2. Using the magic Library**
The `python-magic` library inspects file headers (magic numbers) to identify file types.
```bash
pip install python-magic
```

```python
import magic

file_path = "example.pkl"
file_type = magic.from_file(file_path, mime=True)
print(f"File type: {file_type}")
```
- **Advantage**: Does not rely on file extensions and provides more accurate results by inspecting file content.

**[⬆ back to top](#table-of-contents)**

---

### **3. Reading File Headers Manually**
Many file formats have specific "magic numbers" or headers. You can read the first few bytes to identify the format.
```python
file_signatures = {
    b'\x89PNG': "PNG Image",
    b'\xFF\xD8\xFF': "JPEG Image",
    b'%PDF': "PDF Document",
    b'PK': "ZIP or DOCX",
}

file_path = "example.pdf"
with open(file_path, 'rb') as f:
    header = f.read(4)
    for signature, file_type in file_signatures.items():
        if header.startswith(signature):
            print(f"File type: {file_type}")
            break
    else:
        print("Unknown file type")
```
- **Advantage**: Fully customizable for specific formats.

**[⬆ back to top](#table-of-contents)**

---

### **4. Using the filetype Library**
The `filetype` library provides a simple API for identifying file formats.
```bash
pip install filetype
```

```python
import filetype

file_path = "example.pkl"
kind = filetype.guess(file_path)
if kind:
    print(f"File type: {kind.mime}, Extension: {kind.extension}")
else:
    print("Unknown file type")
```
**[⬆ back to top](#table-of-contents)**

---

### **5. By File Extension (Basic Approach)**
If the extension is reliable, you can use `os.path.splitext`.
```python
import os

file_path = "example.pkl"
extension = os.path.splitext(file_path)[-1]
print(f"File extension: {extension}")
```
- **Limitation**: Depends entirely on the file having a correct extension.

**[⬆ back to top](#table-of-contents)**

---

### Conclusion
- For **extension-based detection**, use `mimetypes` or `os.path.splitext`.
- For **content-based detection**, use `magic`, `filetype`, or inspect file headers manually.
- The choice of method depends on the level of accuracy and the types of files you're working with.

**[⬆ back to top](#table-of-contents)**