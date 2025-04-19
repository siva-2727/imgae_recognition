
# 🛠️ Problem Solving Guide - Python Image Analyzer

This guide helps troubleshoot common issues encountered when running the image analysis script using ResNet50 and BLIP models.

---

## ✅ Dependencies Not Found

**Symptoms:**
- `ModuleNotFoundError: No module named 'torch'`
- `No module named 'transformers'` or similar

**Solution:**
Install all required libraries using:

```bash
pip install torch torchvision transformers Pillow
```

---

## 📥 BLIP Model Loads Slowly or Freezes

**Symptoms:**
- Long wait during model loading
- Script becomes unresponsive

**Solution:**
- Ensure a stable internet connection (models are downloaded from Hugging Face).
- Add `max_new_tokens=20` to limit BLIP output length:

```python
out = caption_model.generate(**inputs, max_new_tokens=20)
```

---

## 📁 FileNotFoundError: imagenet-simple-labels.json

**Symptoms:**
- Script crashes with missing labels file

**Solution:**
Download `imagenet-simple-labels.json` from:
👉 https://github.com/anishathalye/imagenet-simple-labels/blob/master/imagenet-simple-labels.json

Save it in the **same directory** as your Python script.

---

## 🖼️ Image Not Found or Not Displaying

**Symptoms:**
- FileNotFoundError: uploads/pexels-photo-170811.jpeg
- Pillow image errors

**Solution:**
- Ensure the path `uploads/pexels-photo-170811.jpeg` exists.
- Verify the image is in `.jpeg`, `.jpg`, or `.png` format.
- Always convert to RGB:

```python
image = Image.open(image_path).convert("RGB")
```

---

## 🧠 Model Inference Is Too Slow?

**Solution:**
- Use `ResNet18` for faster performance.
- Resize images to smaller dimensions during preprocessing.
- Run on a machine with GPU support if available.

---

## 🧪 Debugging Tips

- Add print statements to trace which step is failing.
- Log `image_path` before using it.
- Catch exceptions using `try-except` blocks for debugging.

---

## 💬 Need Help?

Share:
- The full error message
- Your Python and library versions
- A sample image that caused the issue

---
