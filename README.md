
# Sight & Insights — Repository package (prepared for GitHub)

این مخزن شامل فایل‌های لازم برای بارگذاری کد مقاله "Sight & Insights: Multimodal Retinal AI for Parkinson's Disease" است.
فایل‌ها به‌منظور انتشار در GitHub بسته‌بندی شده‌اند و شامل:
- کد اصلی (اسکلت اولیه): `sight_and_insights.py`
- متن اصلیِ کاملِ اسکریپت (ارجاعی): `original_script_full.txt`
- فایل‌های پیکربندی و توضیحات: `README.md`, `requirements.txt`, `LICENSE`, `.gitignore`
- اسکریپت اجرا: `run.sh`

**توجه مهم:** برای اجرای کامل و آموزش مدل‌ها به منابع محاسباتی (GPU)، داده‌های واقعی و مسیرهای صحیح نیاز دارید. این بسته یک نقطه شروع برای بارگذاری در GitHub است و ممکن است نیاز به اصلاحات جزیی داشته باشد تا با محیط شما سازگار شود.

## ساختار پیشنهادی پوشه‌ها (در محیط اجرا)
```
FUNDUS/
  Healthy/
    <participantID_left_OD.jpg, ...>
  Parkinson/
    <participantID_left_OD.jpg, ...>
Excels/
  final_nodisease.xlsx
  final_parkinson.xlsx
```

## نحوه اجرای سریع (local / Colab)
1. نصب نیازمندی‌ها (پیشنهاد: در یک virtualenv یا Colab):
```bash
pip install -r requirements.txt
```

2. مسیرها را در بالای `sight_and_insights.py` (کلاس `Config`) مطابق با محیط خود تنظیم کنید:
- `IMAGE_DIR` مسیر پوشه تصاویر FUNDUS
- `TABULAR_DIR` مسیر فایل‌های Excel
- `SAVE_DIR` و `XAI_DIR` مسیرهای خروجی

3. اجرای تست اولیه:
```bash
python sight_and_insights.py
```

> اگر می‌خواهید اجرای کالاب را ترجیح دهید، از Google Colab استفاده کنید و پوشه Google Drive را مانت کنید:
> ```py
> from google.colab import drive
> drive.mount('/content/drive')
> ```

## مجوز
این پروژه تحت مجوز MIT عرضه می‌شود. فایل `LICENSE` را ببینید.

---

# Quick English summary

This package is prepared to upload to GitHub as the starting point for the "Sight & Insights" codebase.
It contains a cleaned starter script and auxiliary files (requirements, license, .gitignore). For full training you must:
- provide the dataset (FUNDUS images and Excels),
- set paths in `Config`,
- run on GPU-equipped environment,
- and optionally complete/restore portions of the original script held in `original_script_full.txt`.

If you want, I can:
- finish porting the entire original script into `sight_and_insights.py` (make it runnable),
- create example notebooks and small unit tests,
- or produce a `conda` environment YAML.

فایل زیپِ حاوی این مخزن آماده است؛ لینک دانلود آن در پیام بعدی قرار می‌گیرد.
