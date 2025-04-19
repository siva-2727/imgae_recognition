#  Image Analyzer App

This is an image analysis application built using **PyTorch** and **Hugging Face's BLIP model**. The app performs:
- **Main Subject Detection**: Using ResNet50, it detects the main subject in an image (e.g., "dog", "cat").
- **Scene Description Generation**: Using BLIP, it generates a natural language description of the image scene.

---

##  Features

- Upload an image.
- The app detects the main subject of the image.
- The app generates a scene description using natural language.
  
---

##  Installation & Setup

1. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    or manually:
    ```bash
    pip install torch torchvision transformers Pillow
    ```

2. Download the **imagenet-simple-labels.json** file from:
    [imagenet-simple-labels.json](https://github.com/anishathalye/imagenet-simple-labels/blob/master/imagenet-simple-labels.json)  
    Save it in the same folder as your script.

---

##  Running the App

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open the app in your browser at:  
   [http://localhost:8501](http://localhost:8501)

3. Upload an image to get the analysis.

---

## 🛠️ Troubleshooting & Common Issues

### `ModuleNotFoundError: No module named '...'`
**Solution:**  
Run:
```bash
pip install -r requirements.txt
```

---

###  BLIP model loads slowly or freezes  
**Symptoms:** App is stuck when loading the model.  
**Solution:**
- Ensure a stable internet connection (models are downloaded from Hugging Face).
- Add `max_new_tokens=20` to limit BLIP output length:

```python
out = caption_model.generate(**inputs, max_new_tokens=20)
```

---

### `FileNotFoundError: imagenet-simple-labels.json`  
**Solution:**  
Download from the GitHub repo and place it next to `app.py`.

---

###  Image errors (`OSError`, or black image)  
**Solution:**  
Always convert image to RGB:

```python
image = Image.open(uploaded_file).convert("RGB")
```

---

###  Streamlit doesn’t launch  
**Solution:**
- Try visiting manually: http://localhost:8501
- Or change port:

```bash
streamlit run app.py --server.port 8502
```

---

##  Need Help?

If you're stuck, share the error message and your environment (OS, Python version) with your team or open an issue.

---

